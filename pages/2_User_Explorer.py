import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib # Or pickle, or use xgb.Booster().load_model()
import os

# --- Configuration & Data Loading ---
# @st.cache_data is crucial for performance - it loads data only once
@st.cache_data
def load_data():
    data_path = 'data' # Make sure this path is correct
    print("Loading features data...") # Will only print on first run/cache miss
    try:
        # Load the FINAL features dataframe (15 features + user_id)
        features = pd.read_csv(os.path.join(data_path, 'final_features_15.csv')) # SAVE THIS FROM YOUR NOTEBOOK
    except FileNotFoundError:
        st.error("ERROR: `final_features_15.csv` not found. Please generate and save it first.")
        return None, None, None

    # --- Optional: Load data needed for order history display ---
    print("Loading order history data (optional)...")
    try:
        # Pre-process and save a smaller file if possible
        orders = pd.read_csv(os.path.join(data_path, 'orders.csv'), dtype={'user_id': 'int32', 'order_id': 'int32', 'order_number': 'int16'})
        opp = pd.read_csv(os.path.join(data_path, 'order_products__prior.csv'), dtype={'order_id': 'int32', 'product_id': 'int32'})
        prods = pd.read_csv(os.path.join(data_path, 'products.csv'), dtype={'product_id': 'int32'})
        # Merge product names for display
        order_history_base = opp.merge(prods[['product_id', 'product_name']], on='product_id')
        order_history_base = order_history_base.merge(orders[['order_id', 'user_id', 'order_number']], on='order_id')
        print("Order history base loaded.")
    except FileNotFoundError:
        st.warning("Order/Product files not found, cannot display order history.")
        order_history_base = None

    # Determine valid user ID range
    min_user_id = features['user_id'].min()
    max_user_id = features['user_id'].max()

    return features, order_history_base, min_user_id, max_user_id

@st.cache_resource # Cache the loaded model object
def load_model():
    model_path = 'models/tuned_xgb_final.ubj' # SAVE THIS FROM YOUR NOTEBOOK using joblib or similar
    print("Loading trained XGBoost model...") # Will only print once
    try:
        # model = joblib.load(model_path) # If saved with joblib
        model = xgb.XGBClassifier() # Initialize empty model
        model.load_model(model_path) # If saved using save_model (json or ubj)
        print("Model loaded.")
        # Get expected feature names from the model if possible (good practice)
        try:
             model_feature_names = model.get_booster().feature_names
        except Exception:
             # Fallback: Manually define based on training if needed
             model_feature_names = ['user_total_orders', 'user_avg_days_since_prior', 'user_avg_basket_size', 'user_reorder_ratio', 'user_median_days_since_prior', 'user_std_days_since_prior', 'user_most_frequent_dow', 'user_most_frequent_hour', 'user_total_departments', 'user_total_aisles', 'user_avg_unique_prods_per_order', 'user_total_items_purchased', 'user_reorder_sum', 'days_since_last_order', 'last_order_basket_size']
             print("Warning: Could not get feature names from model, using hardcoded list.")

        return model, model_feature_names
    except (FileNotFoundError, xgb.core.XGBoostError) as e:
        st.error(f"ERROR loading model: {e}. Ensure '{model_path}' exists and is a valid XGBoost model file.")
        return None, None

# --- Load Data and Model (cached) ---
features_df, order_history_df, min_user, max_user = load_data()
model, model_features = load_model()

# --- Streamlit App UI ---
st.header("Part B: Interactive User Explorer")
st.markdown("Enter a User ID to see their features and predicted likelihood of buying a new product in their next order (based on the final Tuned XGBoost model).")

if features_df is not None and model is not None:
    # Get User Input
    user_id_input = st.number_input(
        f"Enter User ID (between {min_user} and {max_user}):",
        min_value=int(min_user),
        max_value=int(max_user),
        value=1, # Default to user 1
        step=1
    )

    analyze_button = st.button("Analyze User")

    if analyze_button:
        st.markdown("---")
        st.subheader(f"Analysis for User ID: {user_id_input}")

        # Find user features
        user_features = features_df[features_df['user_id'] == user_id_input]

        if user_features.empty:
            st.warning("User ID not found in the dataset used for modeling.")
        else:
            # Display User Features
            st.write("**User Features:**")
            # Prepare features for model prediction (ensure correct order and format)
            # Drop user_id and ensure columns match model's expected features
            user_features_for_pred = user_features[model_features].copy()
            st.dataframe(user_features_for_pred)

            # Make Prediction
            try:
                probability_new = model.predict_proba(user_features_for_pred)[0, 1] # Probability of class 1

                # Determine threshold (using the one optimized for Class 0 F1)
                optimal_threshold_c0 = 0.4540 # From previous analysis

                # Classify based on threshold
                if probability_new >= optimal_threshold_c0:
                    prediction_text = "Likely New Product Buyer (Predicted 1)"
                    pred_color = "green"
                else:
                    prediction_text = "Likely Reorderer (Predicted 0)"
                    pred_color = "orange"

                st.write("**Model Prediction:**")
                st.metric("Probability of Buying >=1 New Product", f"{probability_new:.4f}")
                st.markdown(f"Classification (Threshold={optimal_threshold_c0}): **:{pred_color}[{prediction_text}]**")

                # Optional: Display Last N Orders
                if order_history_df is not None:
                    st.write("**Last 5 Prior Orders:**")
                    user_history = order_history_df[order_history_df['user_id'] == user_id_input].copy()
                    if user_history.empty:
                        st.write("No prior order history found for this user.")
                    else:
                        # Get the last 5 unique order numbers
                        last_5_order_nums = sorted(user_history['order_number'].unique())[-5:]
                        last_5_orders_df = user_history[user_history['order_number'].isin(last_5_order_nums)]
                        # Display grouped by order number
                        for order_num in last_5_order_nums:
                            with st.expander(f"Order Number: {order_num}"):
                                products_in_order = last_5_orders_df[last_5_orders_df['order_number'] == order_num]['product_name'].tolist()
                                st.write(", ".join(products_in_order))


            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Data or model failed to load. Cannot run analysis.")