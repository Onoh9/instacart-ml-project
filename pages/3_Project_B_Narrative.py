import streamlit as st
import pandas as pd
# from PIL import Image # Import if/when you use st.image later

st.set_page_config(layout="wide") # Use wide layout for better space

st.markdown("""
# Part B: Predicting Specific New Product Purchases

## Goal

Having explored whether a user is *generally* likely to buy new products (Part A), the goal of Part B is to become more granular and **predict the probability that a specific user (`user_id`) will purchase a specific product (`product_id`) they have *never bought before*, in their next order.**

The ultimate application is to generate a **ranked list of personalized new product recommendations** for each user.

## How Part B Differs from Part A

*   **Focus:** User-Product interaction vs. User-level behavior.
*   **Target:** Probability for *each* potential (User, New Product) pair vs. Single probability for the user's overall order.
*   **Data Structure:** Requires rows representing (User, Product) pairs vs. Rows representing Users.
*   **Features:** Needs User features + **Product features** + **User-Product Interaction features** vs. Only User features.
*   **Output:** A ranked list of recommended *new* products vs. A single prediction about the user.

## High-Level Plan for Part B

We will follow a similar phased approach, but adapted for this user-product level task:

1.  **Phase B.1: Data Generation & Sampling:**
    *   Define positive examples: (User, New Product) pairs that *were* actually purchased in the 'train' order.
    *   Define negative candidates: (User, New Product) pairs that *were not* purchased in the 'train' order.
    *   Implement **negative sampling**: Select a manageable subset of the vast number of negative candidates to create a balanced or representative training dataset.
    *   Construct the final training table: `(user_id, product_id, target, ...features...)`.

2.  **Phase B.2: Feature Engineering (User, Product, Interaction):**
    *   **User Features:** Adapt/reuse aggregate user features from Part A.
    *   **Product Features:** Calculate features describing product characteristics (e.g., overall popularity, reorder rate, aisle/department info).
    *   **User-Product Interaction Features (CRITICAL):** Engineer features capturing the specific relationship between a user and a product (e.g., user's purchase frequency in the product's category, time since last purchase from aisle/dept, etc.).

3.  **Phase B.3: Model Training:**
    *   Train suitable models (e.g., XGBoost, LightGBM) on the generated user-product level data. These models need to learn from the combined user, product, and interaction features.

4.  **Phase B.4: Evaluation:**
    *   Evaluate model performance using standard metrics (AUC).
    *   Crucially, evaluate using **ranking metrics** (e.g., Precision@k, Recall@k, MAP) to assess how well the model ranks the truly purchased new products for each user.

5.  **Phase B.5: Deployment (Streamlit - Page 2 Conceptual):**
    *   Build the interactive "Personalized Recommender" page.
    *   Backend logic will involve: fetching user features, identifying candidate new products, fetching product features, calculating interaction features *on-the-fly* or via precomputation, getting model probabilities, ranking, and displaying results.

**Complexity Note:** Part B is significantly more complex than Part A, particularly regarding data generation/sampling and feature engineering for user-product interactions. Performance (both model accuracy and prediction speed) is also a key challenge.
            """)

st.markdown("""
# Part B - Phase B.1: Data Generation & Sampling

Our goal is to create a dataset where each row represents a potential purchase of a **specific product** by a **specific user** in their 'train' order, labeled with whether that purchase actually happened (target=1) or not (target=0). We only consider products the user had *never* bought before.

**Challenges:**
*   **Positive Examples:** Relatively few - only the new products actually bought in the 'train' orders.
*   **Negative Examples:** Huge number - for each user, there are ~50k products, most of which they didn't buy and hadn't bought before. We cannot use all of them.

**Approach:**
1.  **Identify Positive Examples:** Find all `(user_id, product_id)` pairs where the product was new for the user *and* appeared in their `orders.csv` 'train' order (via `order_products__train.csv`). Assign `target = 1`.
2.  **Identify Negative Candidates:** For each user, find all `product_id`s they have *never* purchased before (across *all* prior orders).
3.  **Filter Negative Candidates:** From the candidates in step 2, *exclude* any products that were actually purchased in the 'train' order (the positive examples).
4.  **Perform Negative Sampling:** From the remaining vast pool of negative candidates for each user, select a manageable number to include in the training data with `target = 0`. The ratio of negative to positive samples is crucial and affects model training. (e.g., 1:1, 5:1, 10:1 negatives to positives).
5.  **Combine:** Create the final DataFrame containing positive and sampled negative examples.
            """)

st.code("""
--- Phase B.1: Data Generation & Sampling (User-by-User) ---
Loading base datasets...
Base datasets loaded successfully.
Identified 555793 positive examples.
Total unique products: 49688
Pre-aggregating prior purchases per user...
Created lookup for user's prior purchased products.
Processing users individually to generate training samples...
Processing Users: 100%|██████████| 131209/131209 [30:04<00:00, 72.69it/s] 
Combining samples from all users...

Phase B.1 Data Generation Complete (User-by-User). Time taken: 1831.54 seconds.

Final Training Data (`final_training_data_part_b`) Head:
   user_id  product_id  target
0   145241      6801.0       0
1   171231     33616.0       0
2   104900     39993.0       1
3   179805     18015.0       0
4   133084     28366.0       0

Final Training Data Shape: (3334758, 3)

Final Training Data Target Distribution:
target
0    0.833333
1    0.166667
Name: proportion, dtype: float64
--------------------------------------------------
Saving intermediate training data structure to data/partB_training_candidates_user_by_user.csv...
Intermediate data saved.
        """)

st.markdown("""
# Part B - Phase B.2: Feature Engineering (User, Product, Interaction)

In Phase B.1, we created our training data structure: `(user_id, product_id, target)`. The target indicates if the user bought that specific *new* product in their 'train' order.

Now, we need to add features to this dataset. Our model needs information about the user, the product, and their specific interaction to predict the target effectively.

**Feature Categories:**

1.  **User Features:** General purchasing habits (similar to Part A, but calculated for *all* relevant users).
    *   Examples: `u_total_orders`, `u_avg_days_since_prior`, `u_avg_basket_size`, `u_reorder_ratio`, `u_total_items`, etc.
2.  **Product Features:** Characteristics of the product itself based on overall purchasing data.
    *   Examples: `p_purchase_count` (popularity), `p_reorder_rate`, `p_avg_add_to_cart_order`, aisle/department info.
3.  **User-Product Interaction Features:** Features capturing the specific historical relationship.
    *   Examples: `uxp_times_bought_before` (should be 0 for this task as we only predict for new products, but useful conceptually), `uxp_reorder_rate_in_user_orders`, `uxp_days_since_last_bought_aisle/dept`, `uxp_order_streak_for_product`.

**Process:**
1. Calculate User features for all relevant users.
2. Calculate Product features for all products.
3. Merge User and Product features onto our `(user_id, product_id, target)` dataset.
4. Engineer and merge User-Product Interaction features (starting simple).
5. Save the final feature-rich training dataset.
            """)

st.markdown("""
# Part B - Phase B.3: Model Training

We have prepared our training dataset (`df_train`) containing `user_id`, `product_id`, `target`, and a set of User, Product, and basic Interaction features.

Now, we will train a model to predict the `target` variable (whether a user bought a specific new product).

**Model Choice: LightGBM**
Given the potentially large size and high imbalance of this dataset, we will use **LightGBM (Light Gradient Boosting Machine)**.
*   **Why LightGBM?** It's another powerful gradient boosting algorithm like XGBoost, but it's known for being significantly **faster** and often more **memory-efficient**, especially on large datasets. It uses techniques like Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). It also handles categorical features well (though we don't have many explicit ones yet) and includes parameters for handling imbalance.

**Process:**
1. Define features (X) and target (y) from our prepared dataset.
2. Split the data into stratified training and testing sets.
3. Initialize and train a LightGBM Classifier, ensuring we account for class imbalance.
            """)

st.markdown("""
# Part B - Phase B.4: Evaluation

Now that we have trained our LightGBM model (`lgbm_model`) on the user-product level data, we need to evaluate its performance on the held-out test set (`X_test_B`, `y_test_B`).

**Evaluation Metrics:**

1.  **AUC-ROC:** As before, this measures the model's overall ability to distinguish between positive (target=1) and negative (target=0) user-product pairs. Higher is better (1.0 is perfect, 0.5 is random).
2.  **Classification Report / Confusion Matrix:** We can still look at these using a standard 0.5 threshold (or another chosen threshold) to understand precision/recall for predicting if *any* specific sampled pair is positive or negative. However, due to the extreme imbalance and sampling, these might be less informative than ranking metrics.
3.  **Precision@k:** This is a crucial **ranking metric** for recommendation tasks. It answers: "If we look at the top *k* products recommended by the model for a user (those with the highest predicted probability), what proportion of those *k* products did the user actually buy?"
    *   We need to calculate this *per user* and then average it.
    *   Requires predicting probabilities for *all* relevant new products for test users, not just the sampled ones used for training/testing the classifier itself. This makes direct calculation complex within this evaluation framework.

**Simplified Evaluation Approach for Now:**
For this iteration, we will focus on:
*   **AUC-ROC** on the test set (`X_test_B`, `y_test_B`) as a measure of overall ranking ability.
*   **Classification Report** using the default 0.5 threshold to get a basic sense of classification performance on the sampled data, keeping the high imbalance in mind.

Calculating true Precision@k would require generating predictions for a much larger set of candidate products per user, which is beyond the scope of this initial modeling step.
            """)

st.subheader("LightGBM Evaluation Metrics, CLass 0(No Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9166") #, delta="0.0198 vs Default XG") # Example delta
col2.metric("Precision (Class 0)", "0.96")
col3.metric("Recall (Class 0)", "0.84")
col4.metric("F1-Score (Class 0)", "0.90")

st.subheader("LightGBM Evaluation Metrics, CLass 1(Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9166") #, delta="0.0198 vs Default XG") # Example delta
col2.metric("Precision (Class 1)", "0.51")
col3.metric("Recall (Class 1)", "0.84")
col4.metric("F1-Score (Class 1)", "0.64")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("LightGBM Confusion Matrix")
        st.image('images/partB_lgbm_cm_05.png', caption='Confusion Matrix for our baseline LightGBM with threshold of 0.5')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_lgbm_cm_05.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("LightGBM Feature Importances")
        st.image('images/partB_lgbm_feat_imp.png', caption='Bar chart of LightGBM Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_lgbm_feat_imp.png' exists.")

st.write("")

st.markdown("""
### Analysis: Initial Part B Model Performance (LightGBM)

We trained a LightGBM classifier on the user-product level data (with ~1:5 negative:positive sampling) and evaluated its performance on the held-out test set.

**Key Findings:**

1.  **Excellent AUC-ROC (0.9166):** This is a very strong AUC score! It indicates the model has a high degree of separability and is very effective at ranking user-product pairs where the user is likely to buy the new product higher than pairs where they are not. This is a promising result for a recommendation-style task.
2.  **Performance at Default Threshold (0.5):**
    *   **Recall (Class 1 - Buy): 0.8381** - The model identifies about 84% of the actual new product purchases within the sampled test set. This is quite good.
    *   **Precision (Class 1 - Buy): 0.5118** - When the model predicts a user *will* buy a specific new product, it's correct only about 51% of the time. This is significantly impacted by the large number of negative samples; even a good model will make many false positive predictions when the base rate of positives is low (~16.7% in our test set).
    *   **F1-Score (Class 1 - Buy): 0.6355** - A reasonable balance for the positive class.
    *   **Class 0 (No Buy) Performance:** Precision is high (0.96) - when it predicts 'No Buy', it's almost always right. Recall is also high (0.84) - it correctly identifies most of the true negative samples *within our test set*.
    *   **Accuracy (0.8398):** Primarily reflects good performance on the majority Class 0.

3.  **Feature Importance (Gain):** *(Updated based on your output)*
    *   **Product features dominate:** The overall popularity (`p_purchase_count`) and reorder characteristics (`p_reorder_rate`) of the *product itself* are by far the most important features according to gain. This makes sense – popular and frequently reordered products are generally more likely to be tried, regardless of the user.
    *   **Interaction matters:** Our single interaction feature (`uxp_dept_purchase_count` - user's history with the product's department) is the third most important feature, highlighting the value of capturing the specific user-product context.
    *   **User features contribute:** User characteristics like average basket size (`u_avg_basket_size`), overall reorder ratio (`u_reorder_ratio`), and total items purchased (`u_total_items_purchased`) also contribute significantly, but less than the top product/interaction features.

**Conclusion & Next Steps for Part B:**

*   **Promising Start:** The LightGBM model shows strong potential, particularly demonstrated by the high AUC score (0.9166), indicating good ranking ability. Product characteristics and user-category interaction are key drivers.
*   **Thresholding Matters:** The default 0.5 threshold yields high recall but mediocre precision for predicting purchases (Class 1). For a real application (like showing top N recommendations), we would likely operate based on the *probability scores* rather than a fixed threshold, or we might tune the threshold differently.
*   **Areas for Improvement:**
    1.  **More Interaction Features:** Adding more features describing the specific user-product relationship (history with aisle, time since last purchase in category, etc.) is the most likely way to further improve AUC and precision, building on the success of `uxp_dept_purchase_count`.
    2.  **Hyperparameter Tuning:** Optimize LightGBM's parameters (`n_estimators`, `learning_rate`, `num_leaves`, regularization params) using `RandomizedSearchCV` or similar, optimizing for AUC.
    3.  **Negative Sampling Strategy:** Experiment with different negative sampling ratios or techniques.
    4.  **Ranking Metric Evaluation:** Implement proper ranking metrics (Precision@k, Recall@k, MAP).

For now, we have successfully built and evaluated a baseline model for Part B! The next logical steps would be **more interaction feature engineering** or **hyperparameter tuning**.
            """)

st.markdown("""
# Part B - Phase B.2 (Continued): Adding More Interaction Features

Our initial LightGBM model showed strong AUC and highlighted the importance of the product itself (`p_purchase_count`, `p_reorder_rate`) and our single interaction feature (`uxp_dept_purchase_count`).

Let's enhance the model's understanding of the user-product relationship by adding more interaction features:

**New Interaction Features:**

1.  **`uxp_aisle_purchase_count`**: How many times has this user bought *any* product from *this specific product's aisle*? (Similar to the department one, but more granular).
2.  **`uxp_user_reorder_rate_in_dept`**: What is the user's *personal* reorder rate for products specifically *within this product's department*? (Does the user tend to reorder things *in this category*?)
3.  **`uxp_user_reorder_rate_in_aisle`**: Same as above, but for the product's specific aisle.

**Process:**
1. Load the intermediate training data structure (`partB_training_candidates...csv`).
2. Load the pre-calculated User Features and Product Features (if saved previously, otherwise recalculate).
3. Merge User and Product features onto the training structure.
4. Calculate the *new* interaction features.
5. Merge the new interaction features.
6. Save the updated, feature-rich dataset.
            """)

st.markdown("""
# Part B - Phase B.3 (Repeated): Model Training with Enhanced Features

We have now added several user-product interaction features (`uxp_aisle_purchase_count`, `uxp_user_reorder_rate_in_dept`, `uxp_user_reorder_rate_in_aisle`) to our user-product level dataset.

We will now retrain our LightGBM model using this updated, feature-rich dataset (`partB_training_data_with_features_v2.csv`). Our hypothesis is that these interaction features, which capture more specific user-category relationships, will provide valuable signals to the model and lead to improved performance (especially in AUC).

**Process:**
1. Load the latest dataset with all features.
2. Define features (X) and target (y).
3. Split into stratified training and testing sets.
4. Initialize and train the LightGBM Classifier (using the same parameters as the previous B.3 run for a direct comparison initially, including imbalance handling).
            """)

st.markdown("""
# Part B - Phase B.4 (Repeated): Evaluation with Enhanced Features

We have retrained the LightGBM model (`lgbm_model_final`) using the dataset enriched with additional user-product interaction features. Now, we evaluate this updated model on the held-out test set (`X_test_B_final`, `y_test_B_final`).

We will focus on the same key metrics as before:
*   **AUC-ROC:** To assess the overall ability to rank positive examples higher than negative ones.
*   **Classification Report / Metrics at 0.5 Threshold:** To understand the precision/recall trade-offs, especially given the imbalance.

We will compare these results directly to the previous LightGBM model (trained on fewer features) to quantify the impact of the added interaction features.
            """)

st.subheader("LightGBM with Enhanced Features Evaluation Metrics, CLass 0(No Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9186" , delta=f"{(0.9186-0.9166):4f} vs Default LightGBM")
col2.metric("Precision (Class 0)", "0.96", delta="0.0 vs baseline LightGBM", delta_color = "off")
col3.metric("Recall (Class 0)", "0.84", delta="0.0 vs baseline LightGBM", delta_color = "off")
col4.metric("F1-Score (Class 0)", "0.90", delta="0.0 vs baseline LightGBM", delta_color = "off")

st.subheader("LightGBM with Enhanced Features Evaluation Metrics, CLass 1(Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9186" , delta=f"{(0.9186-0.9166):4f} vs baseline LightGBM")
col2.metric("Precision (Class 1)", "0.52", delta = f"{(0.52-0.51):4f} vs baseline LightGBM")
col3.metric("Recall (Class 1)", "0.84", delta="0.0 vs baseline LightGBM", delta_color = "off")
col4.metric("F1-Score (Class 1)", "0.64", delta="0.0 vs baseline LightGBM", delta_color = "off")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("LightGBM with Enhanced Features Confusion Matrix")
        st.image('images/partB_lgbm_cm_enhanced.png', caption='Confusion Matrix for our LightGBM with Enhanced Features')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_lgbm_cm_enhanced.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("LightGBM with Enhanced Features Feature Importances")
        st.image('images/partB_lgbm_feat_imp_enhanced.png', caption='Bar chart of LightGBM with Enhanced Features Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_lgbm_feat_imp_enhanced.png' exists.")

st.write("")

st.markdown("""
### Analysis: Part B Model Performance with Enhanced Features

We retrained the LightGBM model using the dataset now enriched with three interaction features (`uxp_aisle_purchase_count`, `uxp_user_reorder_rate_in_dept`, `uxp_user_reorder_rate_in_aisle`). Let's compare the performance to the previous LightGBM model trained without these specific interaction terms:

**Performance Comparison (LightGBM Part B):**

| Metric              | LGBM (Base Feat. B.2) | LGBM (Enhanced Feat. B.2 Cont.) | Change         |
| :------------------ | :-------------------- | :------------------------------ | :------------- |
| **AUC-ROC**         | 0.9166                | **0.9186**                      | **Improved**   |
| Accuracy (Thr 0.5)  | 0.8398                | **0.8416**                      | **Improved**   |
| Precision 1 (Thr 0.5)| 0.5118                | **0.5152**                      | **Improved**   |
| Recall 1 (Thr 0.5)  | 0.8381                | **0.8402**                      | **Improved**   |
| F1-Score 1 (Thr 0.5)| 0.6355                | **0.6387**                      | **Improved**   |
| Precision 0 (Thr 0.5)| 0.96                  | 0.96                            | No Change      |
| Recall 0 (Thr 0.5)  | 0.84                  | 0.84                            | No Change      |
| F1-Score 0 (Thr 0.5)| 0.90                  | 0.90                            | No Change      |

**Key Observations:**

1.  **AUC Improvement:** The AUC-ROC score saw a **small but positive improvement** from 0.9166 to **0.9186**. This indicates the new interaction features added some marginal value to the model's overall ability to rank positive user-product pairs correctly.
2.  **Metrics at Threshold 0.5:** All metrics calculated at the default 0.5 threshold (Accuracy, Precision 1, Recall 1, F1 1) showed **slight improvements**. Performance on Class 0 remained unchanged at this threshold.
3.  **New Feature Importance:**
    *   `p_purchase_count` remains the most dominant feature by gain.
    *   The **new interaction feature `uxp_aisle_purchase_count` is now the second most important feature!** This strongly suggests that knowing how often a user shops in a product's specific aisle is highly predictive.
    *   `p_reorder_rate` (product's overall reorder rate) is third.
    *   The other two new interaction features (`uxp_user_reorder_rate_in_dept` and `uxp_user_reorder_rate_in_aisle`) appear lower down the list but still contribute more than some base user features like `u_median_days_since_prior`.
4.  **Interaction Features are Valuable:** The significant importance of `uxp_aisle_purchase_count` confirms our hypothesis: explicitly adding features that capture the specific user's relationship with the product's category (aisle/department) is beneficial for this prediction task.

**Conclusion for Part B Feature Engineering:**

Adding just three more interaction features yielded further improvement in our Part B model, particularly boosting the AUC score slightly. The high importance of the user-aisle interaction feature strongly validates this direction.

**Next Steps for Part B:**

While the improvement was positive, it was incremental. To potentially achieve larger gains, we could consider:

1.  **More Interaction Features:** Brainstorm and implement features like time since last purchase in aisle/dept, user's average add-to-cart position for the category, etc.
2.  **Hyperparameter Tuning:** Optimize the LightGBM parameters (`n_estimators`, `learning_rate`, `num_leaves`, regularization, etc.) for this specific dataset and feature set using `RandomizedSearchCV`, likely optimizing for AUC.
3.  **Address Precision:** Investigate techniques (different thresholds, possibly different negative sampling) to improve the Precision for Class 1 if generating highly confident recommendations is important.
4.  **Ranking Metrics:** Implement proper ranking metrics (Precision@k, etc.) for a true evaluation of recommendation quality.

Given the strong AUC, **Hyperparameter Tuning** seems like a very worthwhile next step to potentially maximize the performance of the current feature set before adding even more complex features.
            """)

st.markdown("""
# Part B - Phase B.5: Hyperparameter Tuning (LightGBM)

Our LightGBM model trained on the enhanced feature set showed improved AUC (0.9186) compared to the baseline Part B model. The added interaction features, particularly `uxp_aisle_purchase_count`, proved valuable.

Now, we'll attempt to further optimize this model by tuning its hyperparameters using `RandomizedSearchCV`.

**Goal:** Find a combination of LightGBM settings (`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `reg_alpha`, `reg_lambda`, `colsample_bytree`, `subsample`) that potentially increases the AUC score achieved during cross-validation on the training data.

**Process:**
1. Define a parameter grid or distribution to search over.
2. Use `RandomizedSearchCV` with cross-validation on the training set (`X_train_B_final`, `y_train_B_final`).
3. Optimize for the 'roc_auc' metric.
4. Train a final LightGBM model using the best parameters found.
5. Evaluate this tuned model on the held-out test set (`X_test_B_final`, `y_test_B_final`).
            """)

st.subheader("Tuned LightGBM Evaluation Metrics, CLass 0(No Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9190" , delta=f"{(0.9190-0.9186):4f} vs Enhanced LightGBM")
col2.metric("Precision (Class 0)", "0.96", delta="0.0 vs Enhanced LightGBM", delta_color = "off")
col3.metric("Recall (Class 0)", "0.85", delta=f"{(0.85-0.84):4f} vs Enhanced LightGBM", delta_color = "off")
col4.metric("F1-Score (Class 0)", "0.90", delta="0.0 vs Enhanced LightGBM", delta_color = "off")

st.subheader("Tuned LightGBM Evaluation Metrics, CLass 1(Buy)")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.9190" , delta=f"{(0.9190-0.9186):4f} vs Enhanced LightGBM")
col2.metric("Precision (Class 1)", "0.52", delta ="0.0 vs Enhanced LightGBM", delta_color = "off")
col3.metric("Recall (Class 1)", "0.84", delta="0.0 vs Enhanced LightGBM", delta_color = "off")
col4.metric("F1-Score (Class 1)", "0.64", delta="0.0 vs Enhanced LightGBM", delta_color = "off")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("Tuned LightGBM Confusion Matrix")
        st.image('images/partB_lgbm_cm_tuned.png', caption='Confusion Matrix for our Tuned LightGBM')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_lgbm_cm_tuned.png' exists.")

st.write("")

st.markdown("""
### Analysis: Hyperparameter Tuning Results (Part B - LightGBM)

We performed `RandomizedSearchCV` (25 iterations, 3-fold CV) to optimize the LightGBM hyperparameters for the Part B task (predicting specific user-product purchases), aiming to maximize AUC on the training data.

**Key Findings:**

1.  **Best CV Score:** The search found a parameter combination yielding a cross-validated AUC of **0.9188**. This is slightly higher than the AUC of the *untuned* model (0.9186) evaluated on the test set previously, suggesting the search found marginally better parameters according to CV.
2.  **Best Hyperparameters:** The optimal parameters found involved:
    *   A high number of estimators (`n_estimators: 1995`).
    *   A relatively low learning rate (`learning_rate: 0.034`).
    *   Specific settings for tree complexity (`max_depth: 10`, `num_leaves: 71`), regularization (`reg_alpha`, `reg_lambda`), and feature/data sampling (`colsample_bytree`, `subsample`).
3.  **Evaluation on Test Set (Tuned Model):**
    *   **AUC-ROC:** The final tuned model achieved an AUC of **0.9190** on the held-out test set. This is a slight improvement over the untuned model's 0.9186 AUC.
    *   **Metrics at 0.5 Threshold:** Compared to the untuned model:
        *   Accuracy increased slightly (0.8416 -> 0.8442).
        *   Precision (Class 1) increased slightly (0.5152 -> 0.5202).
        *   Recall (Class 1) remained effectively the same (0.8402 -> 0.8374).
        *   F1-Score (Class 1) increased slightly (0.6387 -> 0.6418).
        *   Class 0 metrics remained very similar (Precision ~0.96, Recall ~0.85).

**Conclusion for Tuning:**

Hyperparameter tuning resulted in a **small but positive improvement** in the primary metric (AUC) on the test set. The optimized parameters (particularly more trees with a lower learning rate) refined the model slightly.

While the gains from tuning *after* adding interaction features were modest in this run (AUC improved by +0.0004), it confirms that the model trained with enhanced features was already performing close to its potential *with those specific features*. It also provides a slightly more robust final model.

**Final Part B Model:** The **tuned LightGBM model trained on the 15 features (enhanced user/product + interaction)** represents the best outcome for Part B within the scope of this project, achieving an **AUC of 0.9190**.

---

**(Update Final Summary):**

Now, you should go back to your overall project summary Markdown cell (originally Phase 8 / Cell 20.40 or similar) and update the Part B conclusion to mention the final tuned AUC score:

*Example Update:*
"...We then initiated Part B (predicting specific user-product purchases), generating a training dataset, engineering User/Product/Interaction features, and training a LightGBM model. After hyperparameter tuning, the final model achieved a strong **AUC of 0.9190** on the test set, demonstrating good ranking ability for new product recommendations. Further work on Part B could involve more advanced interaction features or exploring different negative sampling strategies..."

---

This concludes the Part B modeling and tuning! You have successfully built and optimized models for both the user-level prediction (Part A) and the user-product level prediction (Part B).
            """)

st.markdown("""
# Comparative Analysis: Part A vs. Part B Models & Insights

We developed two distinct models addressing related but different questions using the Instacart data:

*   **Part A Model (Tuned XGBoost):** Predicts if a USER will buy *any* new product. Goal: Identify explorer vs. habitual users. (AUC ≈ 0.78)
*   **Part B Model (Tuned LightGBM):** Predicts if a USER will buy a *specific* new product. Goal: Rank potential new products for recommendation. (AUC ≈ 0.92)

Let's compare the insights derived from each:

**1. Feature Importance Contrast:**

*   **Part A (User-Level Prediction):**
    *   Dominated by **User Aggregate Features**, primarily `user_reorder_ratio` (negative correlation with buying new).
    *   Recency features (`last_order_basket_size`, `days_since_last_order`) were also highly important, indicating the last order strongly influences the *next* order's overall exploratory nature.
    *   Average user stats (`u_avg_basket_size`) played a secondary role.
*   **Part B (User-Product Level Prediction):**
    *   Dominated by **Product Features**, especially `p_purchase_count` (product popularity). Popular products are simply more likely to be tried as a first purchase.
    *   **Interaction Features** were highly significant (`uxp_aisle_purchase_count` was #2). Knowing the user's history with the product's *category* is crucial.
    *   **User Aggregate Features** (`u_avg_basket_size`, `u_reorder_ratio`) still contribute significantly but are less dominant than in Part A, indicating that *who the user is* matters, but *what the product is* and the *user-category fit* matters more for specific product prediction.

**2. Model Performance & Behavior:**

*   **Part B's Higher AUC:** The significantly higher AUC in Part B (0.92 vs 0.78) suggests that predicting whether a *specific* popular product will be bought (even by a new user) is an easier task (more signal) than predicting the *general exploratory tendency* of a user across their entire next basket. Product popularity itself provides a strong baseline.
*   **Different Trade-offs:**
    *   Part A XGBoost was tuned/thresholded to achieve high Recall for Class 0 (identifying non-explorers).
    *   Part B LightGBM (at default threshold) achieved high Recall for Class 1 (identifying successful new product purchases *among the sampled pairs*), useful for finding *some* good recommendations, but likely needs threshold tuning depending on the goal (e.g., high precision recommendations).

**3. Complementary Insights:**

*   Part A identifies *which users* are generally open to trying new things.
*   Part B identifies *which specific new things* a user (explorer or not) is most likely to try, heavily influenced by product popularity and category fit.
*   **Synergy:** The Part A prediction (user exploration score) could potentially be used as an *additional feature* in an even more advanced Part B model to further refine recommendations.

**Conclusion:** Both models provide valuable but different perspectives on customer behavior. Part A helps segment users, while Part B enables specific product recommendations. The feature importance differences clearly highlight the shift in focus from overall user history to specific product characteristics and user-category interactions when moving to the more granular prediction task.
            """)

st.markdown("""
# Part B - Phase B.6: Error Analysis (Tuned LightGBM)

We will now analyze the errors made by our best Part B model (the tuned LightGBM using the enhanced 15-feature set). Understanding *which* (User, New Product) pairs it misclassifies can guide future improvements, particularly further interaction feature engineering.

**Goal:** Identify patterns distinguishing misclassified pairs (False Positives, False Negatives) from correctly classified pairs (True Positives, True Negatives).

**Approach:**
1. Use the predictions made on the test set (`X_test_B_final`, `y_test_B_final`) by the tuned LightGBM model (`final_lgbm_model_tuned`). We will use the **default 0.5 threshold** for this analysis, as it provides a reasonable starting point before specific threshold optimization for ranking metrics.
2. Categorize each test sample as TP, TN, FP, or FN.
3. Compare the feature distributions across these categories, focusing on key user, product, and interaction features.
            """)

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("Final Tuned LightGBM Error Analysis")
        st.image('images/partB_error_analysis_dist.png', caption='Error Analysis of our Tuned LightGBM predictors')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/partB_error_analysis_dist.png' exists.")

st.write("")

st.markdown("""
### Analysis: Part B Model Errors (Tuned LightGBM, Threshold=0.5)

We analyzed the characteristics of the (User, New Product) pairs that our tuned LightGBM model misclassified on the test set, using the default 0.5 probability threshold.

**Classification Distribution:**
*   The model correctly identifies most negative examples (True Negatives, TN) and a good portion of positive examples (True Positives, TP).
*   There are still a notable number of False Positives (FP) and fewer, but still present, False Negatives (FN).

**Comparing Misclassified vs. Correctly Classified Pairs:**

1.  **False Positives (FP - Predicted Buy, Actual No Buy):**
    *   *Why did the model predict 'Buy'?* Compared to True Negatives (TN - correctly predicted 'No Buy'), False Positives tend to involve:
        *   **MUCH More Popular Products:** Mean `p_purchase_count` is ~3360 for FP vs. ~113 for TN. The model seems heavily influenced by product popularity, predicting a purchase even when the specific user doesn't bite.
        *   **Higher Product Reorder Rate:** Mean `p_reorder_rate` is ~0.49 for FP vs. ~0.34 for TN. Products that are generally reordered more often seem more likely to be incorrectly predicted as a *new* purchase.
        *   **Slightly More User-Aisle Interaction:** Mean `uxp_aisle_purchase_count` is ~3.3 for FP vs. ~1.1 for TN. The user has slightly more history in the product's aisle for FPs.
        *   **User Features:** User features (like reorder ratio, basket size, history length) are surprisingly **very similar** between FP and TN groups. This suggests the model primarily makes FP errors based on *product characteristics* rather than user characteristics when the threshold is 0.5.
    *   *Conclusion (FP):* The model incorrectly predicts 'Buy' mainly for **popular, frequently reordered products** where the user has *some* minimal history in the aisle, even if the user's overall profile (reorder ratio, etc.) doesn't strongly suggest exploration.

2.  **False Negatives (FN - Predicted No Buy, Actual Buy):**
    *   *Why did the model predict 'No Buy'?* Compared to True Positives (TP - correctly predicted 'Buy'), False Negatives tend to involve:
        *   **MUCH Less Popular Products:** Mean `p_purchase_count` is ~280 for FN vs. ~23,500 for TP! The model struggles to predict purchases for less popular new items.
        *   **Lower Product Reorder Rate:** Mean `p_reorder_rate` is ~0.41 for FN vs. ~0.54 for TP.
        *   **Higher User Reorder Ratio:** Mean `u_reorder_ratio` is ~0.59 for FN vs. ~0.36 for TP. Users who historically reorder more are harder to predict when they *do* try something new.
        *   **Significantly Less User-Aisle Interaction:** Mean `uxp_aisle_purchase_count` is ~1.7 for FN vs. ~6.3 for TP. Lack of user history in the product's specific aisle makes the model hesitant to predict a purchase.
    *   *Conclusion (FN):* The model incorrectly predicts 'No Buy' primarily for **less popular, less reordered new products**, especially when the user is typically a **high reorderer** and has **little or no prior purchase history in that product's specific aisle**.

**Overall Insights from Errors:**

*   **Product Popularity Dominance:** The model relies heavily on overall product popularity (`p_purchase_count`) for its predictions at this threshold. It successfully predicts purchases of popular new items (TP) but often incorrectly predicts purchases of popular items that *aren't* bought (FP). It struggles significantly with less popular items (many FN).
*   **Interaction Feature Value Confirmed:** The difference in `uxp_aisle_purchase_count` between TP/FN and FP/TN highlights its importance. User history *within the category* is a key differentiator.
*   **Need for More Signals:** The difficulty in separating FP from TN (similar user stats) and FN from TP (strong user/product differences) suggests that more nuanced features might be needed. Features capturing *trends* in user behavior, *similarity* between products, or perhaps *time since last category purchase* could potentially help resolve these ambiguities.

**Next Steps Implied:**

While tuning might help, this error analysis strongly reinforces the potential value of **more sophisticated Feature Engineering**, particularly focusing on:
1.  Features that moderate the effect of raw product popularity (e.g., popularity *relative* to its category).
2.  More user-product interaction features (time-based, sequence-based).
3.  Features describing user trends, not just averages.
            """)

