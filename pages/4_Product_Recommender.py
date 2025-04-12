import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle # To load the prior products lookup
import gc
import time

# --- Configuration & Data/Model Loading ---
st.set_page_config(layout="wide")
st.title("🛒 Part B: Personalized New Product Recommender")
st.markdown("Enter a User ID to get recommendations for **specific new products** they might be likely to buy next, based on the **Tuned LightGBM model (Part B)**.")

# @st.cache_data - Load dataframes
@st.cache_data
def load_part_b_assets():
    data_path = 'data'
    print("Loading Part B assets...")
    assets = {}
    try:
        # Load features and set index AFTER loading
        assets['user_features'] = pd.read_csv(os.path.join(data_path, 'partB_user_features.csv'))
        assets['product_features'] = pd.read_csv(os.path.join(data_path, 'partB_product_features.csv'))
        assets['products'] = pd.read_csv(os.path.join(data_path, 'products.csv'), dtype={'product_id': 'int32', 'aisle_id': 'int16', 'department_id': 'int8'})

        # Set index now
        assets['user_features'].set_index('user_id', inplace=True)
        assets['product_features'].set_index('product_id', inplace=True)
        assets['products'].set_index('product_id', inplace=True)

        # Load the lookup dictionary
        lookup_path = os.path.join(data_path, 'user_to_prior_products.pkl')
        with open(lookup_path, 'rb') as f:
            assets['user_prior_prods_lookup'] = pickle.load(f)

        # --- Load Optional Interaction Feature Lookups ---
        # Initialize to None
        assets['user_aisle_counts'] = None
        assets['user_dept_rrates'] = None
        assets['user_aisle_rrates'] = None

        # Try loading each one, set index if loaded
        try:
             df_temp = pd.read_csv(os.path.join(data_path,'partB_user_aisle_counts.csv'))
             assets['user_aisle_counts'] = df_temp.set_index(['user_id', 'aisle_id'])
             print("Loaded user-aisle counts.")
        except FileNotFoundError:
             print("Warning: User-aisle counts file not found. Interaction feature might be missing.")

        try:
             df_temp = pd.read_csv(os.path.join(data_path,'partB_user_dept_rrates.csv'))
             assets['user_dept_rrates'] = df_temp.set_index(['user_id', 'department_id'])
             print("Loaded user-dept reorder rates.")
        except FileNotFoundError:
             print("Warning: User-dept reorder rates file not found. Interaction feature might be missing.")

        try:
             df_temp = pd.read_csv(os.path.join(data_path,'partB_user_aisle_rrates.csv'))
             assets['user_aisle_rrates'] = df_temp.set_index(['user_id', 'aisle_id'])
             print("Loaded user-aisle reorder rates.")
        except FileNotFoundError:
             print("Warning: User-aisle reorder rates file not found. Interaction feature might be missing.")
        # --- End Optional Lookups ---

        print("Part B assets loaded.")
        min_user_id = int(assets['user_features'].index.min())
        max_user_id = int(assets['user_features'].index.max())
        # Get all product IDs from the products table index
        all_product_ids = assets['products'].index.unique().tolist()
        return assets, min_user_id, max_user_id, all_product_ids

    except FileNotFoundError as e:
        st.error(f"ERROR loading essential Part B data: {e}. Please ensure `partB_user_features.csv`, `partB_product_features.csv`, `products.csv`, and `user_to_prior_products.pkl` exist in the 'data' folder.")
        return None, 1, 206209, [] # Default range
    except Exception as e:
        st.error(f"An unexpected error occurred loading Part B assets: {e}")
        return None, 1, 206209, []


@st.cache_resource # Cache the loaded model object
def load_part_b_model():
    model_path = os.path.join('models', 'partB_lgbm_model_v2_tuned.txt') # Tuned LGBM
    print(f"Attempting to load Part B model from: {model_path}")
    try:
        bst = lgb.Booster(model_file=model_path)
        print("Part B model loaded.")
        # Get feature names from booster if possible
        try:
            model_feature_names = bst.feature_name()
            if not model_feature_names: # Handle case where booster might return empty list
                 raise ValueError("Booster returned empty feature name list.")
        except Exception:
             # Fallback: Manually define based on training if needed
             model_feature_names = ['u_total_orders', 'u_avg_days_since_prior', 'u_std_days_since_prior', 'u_median_days_since_prior', 'u_total_items_purchased', 'u_reorder_sum', 'u_reorder_ratio', 'u_avg_basket_size', 'p_purchase_count', 'p_reorder_sum', 'p_avg_add_to_cart', 'p_reorder_rate', 'uxp_aisle_purchase_count', 'uxp_user_reorder_rate_in_dept', 'uxp_user_reorder_rate_in_aisle']
             st.warning("Could not get feature names from model automatically, using hardcoded list.")
        return bst, model_feature_names
    except (FileNotFoundError, lgb.basic.LightGBMError) as e:
        st.error(f"ERROR loading Part B model: {e}. Ensure '{model_path}' exists.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred loading the Part B model: {e}")
        return None, None

# --- Load Assets ---
assets, min_uid_b, max_uid_b, all_prods = load_part_b_assets()
model_b, model_b_features = load_part_b_model()

# --- Streamlit UI ---
if assets is not None and model_b is not None and model_b_features is not None:
    user_id_to_recommend = st.number_input(
        f"Enter User ID for Recommendations (between {min_uid_b} and {max_uid_b}):",
        min_value=min_uid_b,
        max_value=max_uid_b,
        value=1, # Default user
        step=1
    )

    # Option for number of recommendations
    top_n = st.slider("Number of recommendations to show:", min_value=3, max_value=20, value=5)

    recommend_button = st.button("Get New Product Recommendations")

    if recommend_button:
        if user_id_to_recommend not in assets['user_features'].index:
            st.warning(f"User ID {user_id_to_recommend} not found in user features dataset.")
        else:
            st.markdown("---")
            st.subheader(f"Recommendations for User ID: {user_id_to_recommend}")

            start_rec_time = time.time()
            with st.spinner("Generating recommendations... This might take a moment."):
                try:
                    # 1. Get user's prior products
                    prior_prods_set = assets['user_prior_prods_lookup'].get(user_id_to_recommend, set())

                    # 2. Identify candidate NEW products
                    candidate_prods = [p for p in all_prods if p not in prior_prods_set]
                    if not candidate_prods:
                         st.info("This user has already purchased all available products!")
                         st.stop() # Stop execution for this request

                    print(f"Found {len(candidate_prods)} candidate new products for user {user_id_to_recommend}.")

                    # 3. Prepare features for prediction (User + Product + Interaction)
                    # Create DataFrame for prediction
                    predict_df = pd.DataFrame({
                        'user_id': user_id_to_recommend,
                        'product_id': candidate_prods
                    })

                    # Merge User Features (only for this user)
                    user_f = assets['user_features'].loc[[user_id_to_recommend]].reset_index()
                    predict_df = predict_df.merge(user_f, on='user_id', how='left')

                    # --- MODIFIED SECTION for Product Features ---
                    prod_f_lookup = assets['product_features'].reset_index()
                    predict_df = predict_df.merge(prod_f_lookup, on='product_id', how='left')
                    product_feature_cols = [col for col in prod_f_lookup.columns if col.startswith('p_') and col != 'product_id']
                    fill_values_prod = {col: 0 for col in product_feature_cols}
                    predict_df.fillna(value=fill_values_prod, inplace=True)
                    # --- End of MODIFIED SECTION ---

                    # Add aisle/dept from main products df (which has all products)
                    print("DEBUG: Columns in predict_df BEFORE merging products_lookup:", predict_df.columns.tolist())

                    # --- Revised Creation of products_lookup ---
                    # Reset index FIRST to make product_id a column, THEN select columns
                    products_lookup = assets['products'].reset_index()[['product_id', 'aisle_id', 'department_id']]
                    print("DEBUG: Columns in products_lookup:", products_lookup.columns.tolist()) # Should include all 3 now
                    # --- End of Revision ---

                    # Merge candidate products with aisle/dept info
                    predict_df = predict_df.merge(products_lookup, on='product_id', how='left')
                    print("DEBUG: Columns in predict_df AFTER merging products_lookup:", predict_df.columns.tolist()) # CHECK if aisle_id/dept_id appear


                    # Check Null counts AFTER merge:
                    # Ensure columns exist before checking nulls
                    cols_to_check_nulls = ['product_id']
                    if 'aisle_id' in predict_df.columns: cols_to_check_nulls.append('aisle_id')
                    if 'department_id' in predict_df.columns: cols_to_check_nulls.append('department_id')
                    print("DEBUG: Null counts AFTER merge for key columns:\n", predict_df[cols_to_check_nulls].isnull().sum())


                    # Fill potential NaNs for aisle/dept
                    if 'aisle_id' in predict_df.columns:
                        predict_df['aisle_id'].fillna(-1, inplace=True)
                    else:
                        print("WARNING: 'aisle_id' column still not found after merge!")
                        # Add it if absolutely necessary for interaction features, though merge should work
                        # predict_df['aisle_id'] = -1

                    if 'department_id' in predict_df.columns:
                        predict_df['department_id'].fillna(-1, inplace=True)
                    else:
                         print("WARNING: 'department_id' column still not found after merge!")
                         # predict_df['department_id'] = -1


                    # Calculate/Merge Interaction Features (using pre-calculated lookups if possible)
                    print("Calculating/Merging interaction features...")

                    # Create MultiIndex for efficient lookup if lookups were loaded
                    if assets['user_aisle_counts'] is not None:
                        predict_df = predict_df.set_index(['user_id', 'aisle_id'])
                        predict_df['uxp_aisle_purchase_count'] = assets['user_aisle_counts']['uxp_aisle_purchase_count']
                        predict_df = predict_df.reset_index() # Reset index after lookup
                    else:
                        predict_df['uxp_aisle_purchase_count'] = 0

                    if assets['user_dept_rrates'] is not None:
                        predict_df = predict_df.set_index(['user_id', 'department_id'])
                        predict_df['uxp_user_reorder_rate_in_dept'] = assets['user_dept_rrates']['uxp_user_reorder_rate_in_dept']
                        predict_df = predict_df.reset_index()
                    else:
                        predict_df['uxp_user_reorder_rate_in_dept'] = 0

                    if assets['user_aisle_rrates'] is not None:
                        predict_df = predict_df.set_index(['user_id', 'aisle_id'])
                        predict_df['uxp_user_reorder_rate_in_aisle'] = assets['user_aisle_rrates']['uxp_user_reorder_rate_in_aisle']
                        predict_df = predict_df.reset_index()
                    else:
                         predict_df['uxp_user_reorder_rate_in_aisle'] = 0

                    # Fill any NaNs introduced by interaction feature lookups (if index wasn't found)
                    interaction_cols = ['uxp_aisle_purchase_count', 'uxp_user_reorder_rate_in_dept', 'uxp_user_reorder_rate_in_aisle']
                    for col in interaction_cols:
                         if col in predict_df.columns:
                              predict_df[col].fillna(0, inplace=True)

                    # Fill any remaining NaNs from all merges with 0 (catch-all)
                    predict_df.fillna(0, inplace=True)

                    # Ensure features are in the correct order for the model
                    missing_model_features = [f for f in model_b_features if f not in predict_df.columns]
                    if missing_model_features:
                        # Try to recover if only aisle/dept IDs were dropped but model needs them
                        if all(f in ['aisle_id', 'department_id'] for f in missing_model_features):
                             st.warning(f"Model expected {missing_model_features} which were dropped earlier. Adding back with default -1.")
                             if 'aisle_id' not in predict_df.columns: predict_df['aisle_id'] = -1
                             if 'department_id' not in predict_df.columns: predict_df['department_id'] = -1
                             # Recheck missing features
                             missing_model_features = [f for f in model_b_features if f not in predict_df.columns]
                             if missing_model_features: # If still missing others, raise error
                                 raise ValueError(f"Still missing required features after attempting recovery: {missing_model_features}")
                        else:
                             raise ValueError(f"Missing required features after merging: {missing_model_features}")

                    # Reorder columns to match model expectation
                    X_pred_b = predict_df[model_b_features]


                    # 4. Make Predictions
                    print("Predicting probabilities...")
                    probabilities = model_b.predict(X_pred_b) # Use booster's predict

                    # 5. Rank and Display
                    predict_df['predicted_probability'] = probabilities
                    # Merge product names from assets['products']
                    # Ensure assets['products'] still has product_name
                    if 'product_name' not in assets['products'].columns:
                         # If product_name wasn't kept, reload or merge it again
                         products_names_df = pd.read_csv(os.path.join('data', 'products.csv'), usecols=['product_id', 'product_name'], dtype={'product_id':'int32'}).set_index('product_id')
                         recommendations_df = predict_df.merge(products_names_df, left_on='product_id', right_index=True)
                    else:
                         recommendations_df = predict_df.merge(assets['products'][['product_name']], left_on='product_id', right_index=True)


                    # --- Top Recommendations ---
                    top_recommendations = recommendations_df.sort_values(
                        by='predicted_probability', ascending=False
                    ).head(top_n)

                    st.write(f"**Top {top_n} Recommended NEW Products (Most Likely):**")
                    st.dataframe(top_recommendations[['product_id', 'product_name', 'predicted_probability']])


                    # --- Bottom Recommendations ---
                    st.write("") # Add some space
                    bottom_recommendations = recommendations_df.sort_values(
                        by='predicted_probability', ascending=True # Sort ascending for lowest probability
                    ).head(top_n)

                    st.write(f"**Bottom {top_n} Recommended NEW Products (Least Likely):**")
                    st.dataframe(bottom_recommendations[['product_id', 'product_name', 'predicted_probability']])

                    # --- End of Ranking/Display ---

                except Exception as e:
                    st.error(f"An error occurred during recommendation generation: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Print detailed traceback for debugging

            end_rec_time = time.time()
            st.caption(f"Recommendation generation took {end_rec_time - start_rec_time:.2f} seconds.")