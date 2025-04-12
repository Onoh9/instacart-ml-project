import streamlit as st
import pandas as pd
# from PIL import Image # Import if/when you use st.image later

st.set_page_config(layout="wide") # Use wide layout for better space

st.title("Instacart Market Basket Analysis")

# --- Add Buttons Below Title ---

# Define custom CSS for button colors using Markdown
# We'll wrap buttons in divs with specific IDs to target them
# st.markdown("""
# <style>
# /* Style for the container div to center buttons if necessary */
# .button-container {
#     display: flex;
#     justify-content: center;
#     gap: 20px; /* Adds space between buttons */
#     margin-top: 20px;
#     margin-bottom: 30px;
# }

# /* Target the specific divs and style the Streamlit button inside them */
# #button-a .stButton button {
#     background-color: #4CAF50; /* Green */
#     color: white;
#     border-radius: 5px;
#     padding: 10px 24px;
#     border: none;
#     font-weight: bold;
# }

# #button-b .stButton button {
#     background-color: #FF9800; /* Orange */
#     color: white;
#     border-radius: 5px;
#     padding: 10px 24px;
#     border: none;
#     font-weight: bold;
# }

# /* Optional: Hover effects */
# #button-a .stButton button:hover {
#     background-color: #45a049;
# }
# #button-b .stButton button:hover {
#     background-color: #fb8c00;
# }

# </style>
# """, unsafe_allow_html=True)

# # Use columns to help center the buttons horizontally
# # Adjust the numbers in the list for different spacing (left_spacer, content, right_spacer)
# # Using equal spacers on left/right pushes content towards center.
# col1, col2, col3 = st.columns([1, 2, 1]) # Experiment with these ratios, e.g., [1, 1, 1] or [2, 3, 2]

# with col2: # Place buttons in the central column
#     # Use another set of columns *inside* the central one for side-by-side layout
#     b_col1, b_col2 = st.columns(2)

#     with b_col1:
#         # Wrap the button in a div with ID "button-a"
#         st.markdown('<div id="button-a">', unsafe_allow_html=True)
#         # Add the Streamlit button - it won't do anything yet
#         part_a_clicked = st.button("Part A: Project Narrative", use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     with b_col2:
#         # Wrap the button in a div with ID "button-b"
#         st.markdown('<div id="button-b">', unsafe_allow_html=True)
#          # Make this button disabled for now, since Part B isn't built
#         part_b_clicked = st.button("Part B: Product Predictor (Soon!)", disabled=True, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

# # Add a horizontal rule after buttons for separation
# st.markdown("---")










# --- Phase 0 ---
st.header("Phase 1A: Project Introduction & Dataset Deep Dive")
st.subheader("Project Introduction")
st.markdown("""
**Goal:** To predict if an Instacart customer will buy a **new product** (one they haven't purchased previously) in their next order, using their past shopping behavior.

**Application:** This prediction can help Instacart understand customer exploration habits and potentially optimize marketing efforts aimed at introducing new items. It acts as a proxy for identifying customers open to trying something different.
""")

st.subheader("Dataset Overview & Initial Peeks")
st.markdown("""
We are using the public "Instacart Market Basket Analysis" dataset. This dataset describes customer grocery orders over time. It's *relational*, meaning the information is split across multiple files (tables) linked by unique IDs. Understanding these files and their connections is key.

Let's load and inspect each one.
""")

# Example: Displaying info about orders.csv
st.subheader("1. `orders.csv` - The Order Log")
st.markdown("""
*   **Purpose:** Contains metadata about each customer order. The central hub linking users to their order sequence.
*   **Key Columns:** `order_id`, `user_id`, `eval_set` (prior/train/test), `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order`.
""")
# You could show a small static image of the .head() output, or a sample dataframe
# sample_orders_head = pd.read_csv('path_to_saved_head_sample.csv') # Or define manually
# st.dataframe(sample_orders_head)
#st.info("This table tracks individual orders and their timing.") # Use info/success/warning boxes
orders = pd.read_csv('./data/orders.csv')
st.dataframe(orders.head())

st.subheader("2. `products.csv` - The Product Catalog")
st.markdown("""
*   **Purpose:** Lists all unique products available.
*   **Key Columns:** `product_id`, `product_name`, `aisle_id`, `department_id`.
""")
products = pd.read_csv('./data/products.csv')
st.dataframe(products.head())

st.subheader("3. `order_products_prior.csv` & `order-products_train.csv` - Order Contents")
st.markdown("""
*   **Purpose:** Detail exactly *which products* were in *which order*. Connects orders to products.
    *   `__prior.csv`: Contents of past orders (most history). Very large file.
    *   `__train.csv`: Contents of the specific 'train' orders used for defining our target.
*   **Key Columns:** `order_id`, `product_id`, `add_to_cart_order`, `reordered` (1 if bought previously by user, 0 if first time).
""")
order_products_prior = pd.read_csv('./data/order_products__prior.csv')
order_products_train = pd.read_csv('./data/order_products__train.csv')
st.dataframe(order_products_prior.head())
st.dataframe(order_products_train.head())
st.subheader("Reordered Item Distribution (Prior Orders)")
st.markdown("Looking at all items across all prior orders in the dataset:")
st.code("""
# Output from Notebook analysis:
# Value Counts for 'reordered':
# reordered
# 1    0.589697  (~59.0%)
# 0    0.410303  (~41.0%)
# Name: proportion, dtype: float64
""", language=None) # language=None prevents syntax highlighting
st.markdown("This indicates that about **59%** of items purchased in historical orders were items the user had bought before (reordered), while **41%** were first-time purchases for that user.")

st.subheader("4. `aisles.csv` & `departments.csv` - Category Lookups")
st.markdown("""
*   **Purpose:** Translate `aisle_id` and `department_id` into human-readable names.
*   **Columns:** `aisle_id`, `aisle` (name); `department_id`, `department` (name).
            """)
aisles = pd.read_csv('./data/aisles.csv')
departments = pd.read_csv('./data/departments.csv')
st.dataframe(aisles.head())
st.dataframe(departments.head())

st.subheader("How They Connect")
st.markdown("""
*   `orders` links to `order_products__*` via `order_id`.
*   `products` links to `order_products__*` via `product_id`.
*   `products` links to `aisles` and `departments` via their respective IDs.

This structure allows us to trace every item purchased in every order back to the user and the product's details.
            """)


# --- Phase 2 ---
st.header("Phase 2: Defining What We Predict (The Target Variable)")
st.markdown("""

Our goal is to predict if a customer will try a **new product**. Since the dataset doesn't explicitly track marketing interactions, we need a proxy measure based on purchase behavior.

We define our target variable using the `'eval_set'` column in `orders.csv`:
1.  We identify orders marked as `'train'`. These represent the "next order" for a subset of users.
2.  We look at the products purchased in these `'train'` orders (using `order_products__train.csv`).
3.  We compare these products to the customer's *entire* purchase history from *all* their orders marked as `'prior'` (using `orders.csv` and `order_products__prior.csv`).

**Target Variable (`new_product_purchased`):**
*   **1 (Yes):** If the customer's `'train'` order contains at least one product they had **never** purchased in *any* of their `'prior'` orders.
*   **0 (No):** If *all* products in the customer's `'train'` order were products they had purchased at least once in their `'prior'` orders.

This binary variable (0 or 1) will be what our machine learning model tries to predict.

**Output:** We use code to generate a DataFrame named `final_target_df` containing two columns: `user_id` and our calculated `new_product_purchased` target variable for each user included in the 'train' set.
""")

# Display target distribution (maybe as text or a simple bar chart image)
#st.write("Target Variable Distribution:")

st.code("""
# Output from Notebook:
# new_product_purchased
# 1    0.815554  (~81.6%)
# 0    0.184446  (~18.4%)
""", language=None)
#st.markdown("This shows the class imbalance we need to consider.")

# ---- Phase 3 -----

st.markdown("""
## Phase 3: Feature Engineering - Simple Descriptions

### Why Features?

To predict whether a user will buy a new product (`new_product_purchased` = 1 or 0), our machine learning model needs information *about* that user based on their **past behavior**. We can't just feed the raw transaction history into a simple model like Logistic Regression.

Instead, we **engineer features**: descriptive statistics calculated from the user's `'prior'` order history that summarize their typical shopping habits. The model will then learn the relationship between these summary features and the target variable.

### Target Variable Distribution Recap

From Phase 2, we found the distribution of our target variable (`new_product_purchased`):
*   **~81.6%** of users bought at least one new product (value = 1).
*   **~18.4%** of users bought only previously purchased products (value = 0).

This **imbalance** is important. It means most users in our target group *do* try new things. We need to keep this in mind when evaluating our model later – simply guessing "1" for everyone would be ~81.6% accurate but not very useful. 

### Creating Initial Features

We will now calculate a few basic features for each user, using **only their order history marked as `'prior'`**:

1.  `user_total_orders`: Total number of prior orders placed. (Indicates experience/history length).
2.  `user_avg_days_since_prior`: Average number of days between prior orders. (Indicates typical purchase frequency).
3.  `user_avg_basket_size`: Average number of items purchased per prior order. (Indicates typical order size).
4.  `user_reorder_ratio`: Overall proportion of items purchased across all prior orders that were reorders. (Indicates tendency to stick to known items vs. explore).

These features provide a simple numerical profile of each user's past behavior.
            """)

st.code("""
Final Features DataFrame (`features_df`) Head:
   user_id  user_total_orders  user_avg_days_since_prior  user_avg_basket_size  user_reorder_ratio
0        1                 10                  19.555555              5.900000            0.694915
1        2                 14                  15.230769             13.928571            0.476923
2        5                  4                  13.333333              9.250000            0.378378
3        7                 20                  10.684211             10.300000            0.669903
4        8                  3                  30.000000             16.333333            0.265306
                """)

st.markdown("""
# Phase 4: Preparing Data for Modeling

Now that we have:
1.  `final_target_df`: Contains the `user_id` and the target variable (`new_product_purchased`) we want to predict.
2.  `features_df`: Contains the `user_id` and the descriptive features calculated from past behavior.

We need to combine these and prepare them for the machine learning model. This involves two main steps:

1.  **Merging:** Combine the features and the target variable into a single dataset aligned by `user_id`.
2.  **Splitting:** Divide this combined dataset into a **Training Set** and a **Testing Set**.
    *   **Why Split?** We train the model on the Training Set and evaluate its performance on the unseen Testing Set. This tells us how well the model generalizes to new data it hasn't learned from directly, helping us avoid overfitting (where the model just memorizes the training data).
    *   **Why Stratify?** As noted before, our target variable is imbalanced (~81.6% class 1, ~18.4% class 0). Stratified splitting ensures both the training and testing sets maintain this same proportion, leading to a more reliable evaluation.
            """)

st.markdown("""
# Phase 5: Building and Evaluating a Baseline Model

Now we'll build our first predictive model. We start with a simple but effective algorithm called **Logistic Regression**.

## Logistic Regression

*   **What it is:** A statistical method used for binary classification (predicting one of two outcomes, like our 0 or 1).
*   **How it works (simplified):** It learns a linear relationship between the input features (`user_total_orders`, `user_avg_basket_size`, etc.) and the *log-odds* of the target outcome (`new_product_purchased` = 1). It then transforms this log-odds score into a probability (between 0 and 1). By default, if the probability is >= 0.5, it predicts 1; otherwise, it predicts 0.
*   **Why start here?** It's computationally efficient, relatively easy to interpret (we can look at feature coefficients), and provides a crucial **baseline performance**. We need to know how well a simple model does before trying more complex ones.

## How We Evaluate Performance

Since our data is imbalanced (more 1s than 0s), accuracy alone isn't sufficient. We'll use several metrics:

*   **Accuracy:** Overall percentage of correct predictions. (Can be misleading here since we'd get 81% accuracy from guessing yes on all orders, regardless on if we used a model or not.)
*   **Precision (Specificity):** For a specific class (e.g., class 1), answers: "Of all the times the model predicted this class, how often was it right?" (High precision means few False Positives).
*   **Recall (Sensitivity):** For a specific class, answers: "Of all the actual instances of this class, how many did the model correctly identify?" (High recall means few False Negatives).
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both. Useful for comparing overall performance per class.
*   **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between the positive (1) and negative (0) classes across *all* possible probability thresholds. Ranges from 0.5 (random guessing) to 1.0 (perfect separation). This is often a key metric for imbalanced data.
*   **Confusion Matrix:** A table showing the counts of:
    *   True Positives (Actual 1, Predicted 1)
    *   True Negatives (Actual 0, Predicted 0)
    *   False Positives (Actual 0, Predicted 1 - Type I error)
    *   False Negatives (Actual 1, Predicted 0 - Type II error)
            """)


st.subheader("Logistic Regression Evaluation Metrics")
# You can use st.metric for key results
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7523") #, delta="0.0198 vs Default XG") # Example delta
col2.metric("Precision (Class 0)", "0.67")
col3.metric("Recall (Class 0)", "0.16")
col4.metric("F1-Score (Class 0)", "0.26")


# Create columns: [spacer_left, content, spacer_right]
# Adjust the ratios (e.g., [1, 2, 1]) to control centering/width
col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1]) # Middle column is 3x wider than spacers

with col3: # Use the middle column
    try:
        st.subheader("Logistic Regression Confusion Matrix")
        st.image('images/LR_1_Confusion_Matrix.png', caption='Confusion Matrix for our baseline Logistic Regression')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/LR_1_Confusion_Matrix.png' exists.")

# Add some vertical space after if needed
st.write("") # Creates a small gap


st.markdown("""
### Analysis of Logistic Regression Results

The evaluation metrics for our baseline Logistic Regression model reveal several important points:

1.  **Overall Discriminatory Power (AUC-ROC = 0.7523):** The AUC score is significantly above 0.5, indicating that the model has a reasonable ability to differentiate between users who will buy a new product versus those who won't. It's performing much better than random chance.

2.  **High Accuracy, Misleading Picture (Accuracy = 0.8306):** While accuracy seems high, it's heavily influenced by the model correctly predicting the majority class (Class 1: New Product Purchased), which makes up ~81.6% of the data. It doesn't tell the full story about performance on both classes.

3.  **Performance on Majority Class (Class 1 - New):**
    *   The model achieves high Recall (0.9813) and good Precision (0.8385) for Class 1. This means it correctly identifies almost all users who *do* buy a new product, and when it predicts they will, it's often right. The F1-score (0.9043) reflects this strong performance.

4.  **Performance on Minority Class (Class 0 - No New):**
    *   **This is the main weakness.** The Recall for Class 0 is very low (0.16). This means the model only successfully identifies 16% of the users who *did not* buy a new product. It misses the vast majority (84%) of this group.
    *   The Precision for Class 0 (0.67) is better, meaning when it *does* predict Class 0, it's correct about two-thirds of the time. However, it makes this prediction very rarely due to the low recall.
    *   The resulting F1-score (0.26) is poor, highlighting the model's difficulty with this minority class.

5.  **Key Feature Insights (Coefficients):**
    *   The model heavily relies on `user_reorder_ratio` (coefficient -4.05). A higher historical reorder rate strongly predicts *against* buying a new product. This aligns perfectly with intuition.
    *   `user_avg_basket_size` has a moderate positive influence (coefficient 0.13), suggesting larger average orders are slightly associated with buying new products.
    *   The other features (`user_avg_days_since_prior`, `user_total_orders`) have a much smaller linear influence according to this model.

### **Conclusion for Baseline:**

The Logistic Regression model provides a useful, interpretable baseline. It confirms the importance of historical reordering behavior. However, its practical value might be limited by its inability to effectively identify the smaller group of customers who stick only to familiar products (low recall for Class 0). This suggests room for improvement, potentially using models better suited to handling imbalance or capturing more complex relationships.
            """)

st.markdown("""
## Potential Next Steps

Based on these results, several avenues could be explored to potentially improve the model and gain further insights:

1.  **Try More Complex Models:** Algorithms like **Random Forest** or **XGBoost** might capture more complex patterns and interactions between features that Logistic Regression misses. They also have built-in mechanisms (like `class_weight` or `scale_pos_weight`) that could potentially handle the class imbalance more effectively and improve recall for the minority class.
2.  **More Feature Engineering:** Our current model only uses 4 basic features. Creating more features (e.g., related to purchase timing, category preferences, user tenure, characteristics of the *last* order) could provide the model with richer information and improve predictive power.
3.  **Hyperparameter Tuning:** The models (including Logistic Regression) have settings (hyperparameters) that we left at their defaults. Systematically testing different settings (e.g., using GridSearchCV) could optimize performance for our specific dataset and evaluation metric (like AUC or F1-score for Class 0).
4.  **Address Imbalance Differently:** Explore techniques like resampling (e.g., SMOTE to oversample the minority class or random undersampling of the majority class) during training, although careful evaluation is needed.
5.  **Interpretability vs. Performance:** Decide if the slightly lower interpretability of tree-based models (like RF/XGBoost) compared to Logistic Regression coefficients is an acceptable trade-off for potentially higher performance.

For this project's current scope, evaluating a more complex model like Random Forest or XGBoost (Option 1) would be a logical next step to see if we can improve upon this baseline, particularly for the minority class prediction.
            """)

st.markdown("""
### Trying a More Complex Model: Random Forest

Our baseline Logistic Regression model showed decent overall performance (AUC ~0.75) but struggled to identify the minority class (users *not* buying new products). Logistic Regression assumes linear relationships between features and the outcome's probability. Real-world behavior is often more complex.

To potentially capture non-linear patterns and interactions between our features, we'll now try a **Random Forest Classifier**.

**Why Random Forest?**
*   **Handles Non-Linearity:** It builds multiple decision trees, which can naturally model non-linear relationships (e.g., the effect of basket size might plateau).
*   **Captures Interactions:** By design, it considers how different features work together when making splits in the trees.
*   **Robustness:** Generally less sensitive to feature scaling and outliers compared to linear models.
*   **Imbalance Handling:** Includes options like `class_weight='balanced'` which explicitly tells the model to pay more attention to the minority class during training, potentially improving its recall.

We will train it on the same data and evaluate using the same metrics to directly compare its performance to the Logistic Regression baseline.

### A Note on Class Imbalance and `class_weight`

Looking back at our target variable distribution (Phase 2), we saw significant **class imbalance**:

*   Class 1 (Bought New Product): ~81.6%
*   Class 0 (Did Not Buy New Product): ~18.4%

**Why is this important?** Standard machine learning algorithms often aim to minimize the overall number of mistakes. With imbalanced data, a model can achieve high *overall accuracy* simply by mostly predicting the majority class (Class 1 in our case). This can result in a model that is poor at identifying the minority class (Class 0), which might be the group we are particularly interested in understanding or targeting differently. Our Logistic Regression results showed this exact issue (very low Recall for Class 0).

**How `class_weight` Helps:**
The `class_weight` parameter, available in many Scikit-learn models like `RandomForestClassifier`, allows us to address this. It adjusts how much the model gets "penalized" for making mistakes on different classes during training.

*   **`class_weight='balanced'`:** We will use this setting. It automatically assigns higher weights to the minority class (Class 0) and lower weights to the majority class (Class 1). The weights are inversely proportional to how frequent each class is.
*   **Effect:** This tells the algorithm: "Pay more attention to getting the Class 0 predictions correct, even if it means making slightly more errors on the common Class 1." This often helps to **improve the model's ability to identify the minority class (increasing Recall for Class 0)**, potentially at the cost of slightly lower overall accuracy or precision.

Using `class_weight='balanced'` is a common strategy to encourage models to learn more effectively from imbalanced datasets.
            """)

st.subheader("Random Forest Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7031", delta=f"{(0.7031-0.7523):.4f} vs Default LR")
col2.metric("Precision (Class 0)", "0.48", delta=f"{(0.48-.067):.4f} vs Default LR")
col3.metric("Recall (Class 0)", "0.23", delta=f"{(0.23 - 0.16):.4f} vs Default LR")
col4.metric("F1-Score", "0.31", delta=f"{(.31-.26):.4f} vs Default LR")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("Random Forest Confusion Matrix")
        st.image('images/Random_Forest_Confusion_Matrix.png', caption='Confusion Matrix for our baseline Random Forest')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/Random_Forest_Confusion_Matrix.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("Random Forest Feature Importances")
        st.image('images/Random_Forest_Feature_Importances.png', caption='Bar chart of Random Forest Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/Random_Forest_Feature_Importances.png' exists.")

st.write("")

st.markdown("""
### Analysis of Random Forest Results & Next Steps

We trained a Random Forest Classifier, using `class_weight='balanced'` to specifically address the poor performance on the minority class (Class 0 - No New Product) observed with Logistic Regression. Let's compare:

**Key Observations:**

1.  **Minority Class Improvement:** The Random Forest, aided by `class_weight='balanced'`, did successfully **improve Recall for Class 0** (from 0.16 to 0.23). It identified more of the users who *did not* buy new products. The F1-score for Class 0 also slightly increased.
2.  **Overall Performance Decline:** This improvement came at a cost. The **AUC dropped significantly** (from 0.7523 to 0.7031), indicating the model's overall ability to distinguish between the classes worsened. Accuracy and performance on the majority class (Class 1) also slightly decreased.
3.  **Feature Usage:** Random Forest seems to distribute importance more evenly across the top features (`user_reorder_ratio`, `user_avg_basket_size`, `user_avg_days_since_prior`) compared to Logistic Regression, suggesting it leverages them differently.

**Decision & Path Forward:**

While Random Forest addressed the specific goal of improving minority class recall, the drop in overall discriminatory power (AUC) is concerning. It suggests that simply applying a standard non-linear model with basic class weighting wasn't sufficient with our current features.

There are two primary paths forward:

*   **Path A: Try a Different Algorithm:** Explore another powerful algorithm like **XGBoost (Extreme Gradient Boosting)**. XGBoost often achieves state-of-the-art results on tabular data and handles feature interactions and non-linearities differently than Random Forest. It also has robust mechanisms for handling class imbalance (`scale_pos_weight`). We can see if it offers a better balance between overall performance (AUC) and minority class identification.
*   **Path B: More Feature Engineering:** Acknowledge that our current 4 features might be too simple. We could pause modeling and engineer more descriptive features (related to timing, categories, user tenure, etc.) hoping to provide *any* model with more signal to work with.

**Our Chosen Next Step: Try XGBoost (Path A)**

We will proceed with **Path A** for now and evaluate XGBoost. It represents a different type of gradient-boosted ensemble model and is worth testing before concluding that our features are insufficient.

*   **Rationale:** It's valuable to see if a different algorithmic approach can better utilize the existing features.
*   **Contingency:** If XGBoost *also* fails to significantly improve upon the Logistic Regression baseline (especially in AUC), it will strengthen the argument for needing **more feature engineering (Path B)** as the most critical next step.

We will now train and evaluate an XGBoost model, again using appropriate methods to handle class imbalance.
            """)

st.markdown("""
### Trying Another Advanced Model: XGBoost

Following our analysis, the Random Forest model improved minority class recall but decreased overall performance (AUC) compared to the Logistic Regression baseline. Before concluding that we need more features, we'll try one more advanced algorithm: **XGBoost (Extreme Gradient Boosting)**.

**Why XGBoost?**
*   **Different Ensemble Method:** While also using decision trees, XGBoost builds them sequentially. Each new tree tries to correct the errors made by the previous ones (this is "boosting"). This often leads to highly accurate models.
*   **Regularization:** XGBoost includes built-in regularization techniques (L1 and L2) which help prevent overfitting, potentially leading to better generalization than a standard Random Forest on some datasets.
*   **Efficiency:** It's known for its computational efficiency and speed, often leveraging optimized algorithms and parallel processing well.
*   **Imbalance Handling:** Like Random Forest, it can handle imbalance. We will use its specific parameter `scale_pos_weight` which adjusts the weight given to the positive class errors (calculated based on the ratio of negative to positive samples).

By comparing XGBoost's results to both Logistic Regression and Random Forest, we'll get a clearer picture of whether algorithmic changes alone can significantly improve performance with our current features.
            """)

st.subheader("XGBoost Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7483", delta=f"{(0.7483-0.7523):.4f} vs Default LR")
col2.metric("Precision (Class 0)", "0.35", delta=f"{(0.35-.067):.4f} vs Default LR")
col3.metric("Recall (Class 0)", "0.64", delta=f"{(0.64 - 0.16):.4f} vs Default LR")
col4.metric("F1-Score", "0.45", delta=f"{(.45-.26):.4f} vs Default LR")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("XGBoost Confusion Matrix")
        st.image('images/XGB_Confusion_Matrix.png', caption='Confusion Matrix for our baseline XGB')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/XGB_Confusion_Matrix.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("XGBoost Feature Importances")
        st.image('images/XGB_Feature_Importances.png', caption='Bar chart of XGBoost Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/XGB_Feature_Importances.png' exists.")

st.write("")

st.markdown("""
### Analysis of XGBoost Results & Model Comparison

We trained an XGBoost classifier, using `scale_pos_weight` to handle the class imbalance. Here's how it performed compared to Logistic Regression and Random Forest:

| Metric        | Logistic Regression | Random Forest (`balanced`) | XGBoost (`scale_pos_weight`) |
| :------------ | :------------------ | :------------------------- | :------------------------- |
| **AUC-ROC**   | **0.7523**          | 0.7031                     | 0.7483                     |
| Accuracy      | **0.8306**          | 0.8113                     | 0.7123                     |
| Precision 0   | **0.67**            | 0.48                       | 0.35                       |
| **Recall 0**    | 0.16                | 0.23                       | **0.64**                   |
| **F1-Score 0**  | 0.26                | 0.31                       | **0.45**                   |
| Precision 1   | 0.8385              | 0.8435                     | **0.9001**                 |
| Recall 1      | **0.9813**          | 0.9437                     | 0.7280                     |
| F1-Score 1    | **0.9043**          | 0.8908                     | 0.8049                     |

**Key Observations for XGBoost:**

1.  **Minority Class Identification:** XGBoost achieved **by far the highest Recall (0.64) and F1-Score (0.45) for Class 0 (No New Product)**. Using `scale_pos_weight` was very effective in forcing the model to identify a majority (64%) of these difficult-to-find users.
2.  **Trade-off:** This came at the cost of significantly lower Recall (0.7280) for the majority Class 1 compared to the other models. Precision for Class 0 was also the lowest (0.35), meaning many of its Class 0 predictions were incorrect (False Positives). Overall accuracy (0.7123) was the lowest due to the shift in focus towards Class 0.
3.  **AUC Performance:** XGBoost's AUC (0.7483) is very close to Logistic Regression's (0.7523) and much better than Random Forest's (0.7031). This suggests its overall ability to rank users correctly is competitive with the baseline linear model, despite its very different internal mechanics and class performance balance.
4.  **Feature Importance:** Similar to the other models, `user_reorder_ratio` and `user_avg_basket_size` are overwhelmingly the most important features. XGBoost relies even more heavily on these top two compared to Random Forest.

**Overall Comparison & Path Forward:**

*   **No Single Best:** None of the models is universally superior across all metrics.
    *   **Logistic Regression:** Offers the best AUC (slightly) and balanced performance *for the majority class*. It's simple and interpretable but fails on minority recall.
    *   **Random Forest:** In this configuration, it underperformed on AUC and didn't offer compelling advantages.
    *   **XGBoost:** Provides the best performance by far on the **minority class (Recall 0, F1 0)**, making it potentially valuable if identifying users who *don't* try new things is a key goal. Its AUC is competitive with Logistic Regression.
*   **The Feature Ceiling:** The fact that neither RF nor XGBoost significantly surpassed the baseline Logistic Regression's AUC, despite their ability to model complexity, **strongly suggests that we may be limited by our current set of 4 features.** They might not contain enough predictive signal to push performance much higher, regardless of the algorithm.

**Decision & Recommended Next Step:**

Given these findings, the most promising path to achieving a potentially *significant* improvement and building a more impressive project is likely **Phase 3b: More Feature Engineering**.

*   **Rationale:** Adding more diverse and informative features (e.g., related to timing, user tenure, category affinities) has a high potential to boost the performance of *all* model types, potentially lifting the AUC ceiling we seem to be hitting.
*   **Plan:** We will go back and create a richer set of features based on the user's prior history. Afterwards, we can retrain and re-evaluate our models (perhaps focusing on Logistic Regression and XGBoost, as they showed the most promise in different areas) on this expanded feature set. This iterative process of feature engineering and modeling is central to practical machine learning.

We will now proceed to define and calculate additional features.
            """)

st.markdown("""
# Phase 3b: More Feature Engineering

Our initial modeling showed that while algorithms handle the data differently, performance (especially AUC) didn't dramatically improve beyond the baseline. This strongly suggests our initial 4 features might not be capturing enough complexity about user behavior.

Therefore, we will now engineer additional features based on users' `'prior'` order history to provide richer information to the models.

**New Feature Categories:**

1.  **User Timing Patterns:** Look more closely at *when* users shop.
    *   `user_median_days_since_prior`: Median days between orders (more robust to outliers than the average).
    *   `user_std_days_since_prior`: Standard deviation of days between orders (measures consistency of timing).
    *   `user_most_frequent_dow`: Most common day of the week for orders.
    *   `user_most_frequent_hour`: Most common hour of the day for orders.
2.  **User Product Diversity:** Measure the variety of products purchased.
    *   `user_total_departments`: Number of distinct departments purchased from.
    *   `user_total_aisles`: Number of distinct aisles purchased from.
    *   `user_avg_unique_prods_per_order`: Average number of *unique* products per order basket.
3.  **User Reorder Details:** Add more context to reordering.
    *   `user_total_items_purchased`: Total number of items (lines in order_products) across all prior orders.
    *   `user_reorder_sum`: Total count of items flagged as 'reordered' across all prior orders.

We will calculate these and add them to our existing `features_df`.
            """)

st.markdown("""
# Phase 4 (Repeated): Preparing Updated Data for Modeling

We have successfully engineered additional features, expanding our `features_df` from 4 features to 13.

Now, we need to repeat the data preparation steps using this **updated feature set**:

1.  **Merging:** Combine the *new* `features_df` (with 13 features) and the `final_target_df` (target variable).
2.  **Splitting:** Divide this updated combined dataset into new Training and Testing sets (`X_train`, `X_test`, `y_train`, `y_test`). We will use the same `test_size` (20%) and `random_state` (42) as before for consistency and comparability, and crucially, continue to `stratify` by the target variable `y`.

This will give us the necessary inputs, now based on the richer feature set, for retraining our models.
            """)

st.markdown("""
# Phase 5 (Repeated): Evaluating Models with Updated Features

Now that we have engineered a richer set of 13 features, we will retrain and re-evaluate our models to see if the additional information improves predictive performance.

We'll start again with our baseline: **Logistic Regression**. We use the *exact same* modeling code but train and test it on the new data splits (`X_train_upd`, `y_train_upd`, `X_test_upd`, `y_test_upd`) derived from the expanded feature set. We will compare the results directly to the previous Phase 5 results (which used only 4 features).
            """)

st.subheader("Logistic Regression with Updated Features Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7580", delta=f"{(0.7580-0.7523):.4f} vs Default LR")
col2.metric("Precision (Class 0)", "0.65", delta=f"{(0.65-.067):.4f} vs Default LR")
col3.metric("Recall (Class 0)", "0.18", delta=f"{(0.18 - 0.16):.4f} vs Default LR")
col4.metric("F1-Score", "0.28", delta=f"{(.28-.26):.4f} vs Default LR")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("Logistic Regression with Updated Features Confusion Matrix")
        st.image('images/LR2_Confusion_Matrix.png', caption='Confusion Matrix for our updated Logistic Regression')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/LR2_Confusion_Matrix.png' exists.")

st.write("")

st.markdown("""
### Analysis: Logistic Regression with Updated Features

We retrained the Logistic Regression model using the expanded set of 13 features. Let's compare the key metrics to the baseline model which used only 4 features:

| Metric        | LR (4 Features) | LR (13 Features) | Change        |
| :------------ | :-------------- | :--------------- | :------------ |
| **AUC-ROC**   | 0.7523          | **0.7580**       | **Improved**  |
| Accuracy      | 0.8306          | **0.8308**       | Marginal Imp. |
| Precision 0   | 0.67            | 0.65             | Worse         |
| **Recall 0**    | 0.16            | **0.18**         | **Improved**  |
| **F1-Score 0**  | 0.26            | **0.28**         | **Improved**  |
| Precision 1   | 0.8385          | **0.8409**       | Improved      |
| Recall 1      | **0.9813**      | 0.9775           | Worse         |
| F1-Score 1    | **0.9043**      | 0.9041           | Marginal Dec. |

**Key Observations:**

1.  **AUC Improvement:** The AUC-ROC score **improved slightly** from 0.7523 to 0.7580. This indicates the additional features provided *some* extra signal that helps the model better distinguish between the classes overall.
2.  **Minority Class Improvement:** The **Recall for Class 0 improved** from 0.16 to 0.18, and the corresponding F1-score improved from 0.26 to 0.28. While still very low, the model is identifying slightly more of the users who *don't* buy new products. This is a positive sign.
3.  **Other Metrics:** Changes in Accuracy, Precision (Class 1), and F1-score (Class 1) were marginal. There was a slight decrease in Recall for Class 1, likely a consequence of the model trying slightly harder to identify Class 0.
4.  **Updated Feature Importance (Coefficients):**
    *   `user_reorder_ratio` remains the most dominant feature, although its absolute coefficient decreased (-3.46 vs -4.05), suggesting its influence is shared slightly more with other features now.
    *   New features like `user_total_departments` (positive coef: higher variety -> more likely new purchase?), `user_avg_unique_prods_per_order` (positive), and `user_most_frequent_dow` (negative?) now appear among the more influential features, although their coefficients are much smaller than `user_reorder_ratio`.
    *   The coefficients for the original features (`user_avg_basket_size`, `user_avg_days_since_prior`, `user_total_orders`) have changed slightly, reflecting the presence of the new variables in the model.

**Conclusion:**

Adding the extra features provided a **modest but measurable improvement** to the Logistic Regression model, primarily seen in the slight increase in AUC and the slightly better identification of the minority class (Recall 0 and F1 0). The model is leveraging some of the new information.

However, the core challenge remains: **Recall for Class 0 is still very low (0.18).**

**Next Step:** Since the added features provided *some* benefit, it's worth seeing how a more complex model like **XGBoost** performs with this richer feature set. XGBoost might be better equipped to exploit subtle patterns and interactions within these 13 features, potentially leading to a more significant improvement, especially for the minority class.

We will now retrain and evaluate XGBoost using the updated features (`X_train_upd`, `y_train_upd`, etc.).
            """)

st.markdown("""
### Retraining XGBoost with Updated Features

The updated features provided a slight boost to the Logistic Regression model, particularly improving the AUC and the minority class recall a little.

Now, let's see if **XGBoost**, which previously showed the best performance on the minority class (Class 0) with the original 4 features, can leverage this richer set of 13 features more effectively. We hypothesize that XGBoost's ability to model non-linearities and feature interactions might lead to more significant gains with the expanded feature set compared to Logistic Regression.

We will use the same XGBoost setup as before (including `scale_pos_weight` for imbalance) but train and test on the updated data splits (`X_train_upd`, `y_train_upd`, `X_test_upd`, `y_test_upd`).
            """)

st.subheader("XGBoost with Updated Features Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7458", delta=f"{(0.7458-0.7483):.4f} vs Default XGB")
col2.metric("Precision (Class 0)", "0.35", delta=f"{(0.35-0.35):.4f} vs Default XGB")
col3.metric("Recall (Class 0)", "0.63", delta=f"{(0.63 - 0.64):.4f} vs Default XGB")
col4.metric("F1-Score", "0.45", delta=f"{(.45-.45):.4f} vs Default XGB")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("XGBoost with Updated Features Confusion Matrix")
        st.image('images/XGB2_Confusion_Matrix.png', caption='Confusion Matrix for our updated XGB')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/XGB2_Confusion_Matrix.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("XGBoost with Updated Features Feature Importances")
        st.image('images/XGB2_Feature_Importances.png', caption='Bar chart of updated XGBoost Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/XGB2_Feature_Importances.png' exists.")

st.write("")

st.markdown("""
### Analysis: XGBoost with Updated Features & Final Comparison

We retrained the XGBoost model using the expanded set of 13 features, including `scale_pos_weight` to address imbalance. Here's a comparison across all our experiments:

**Model Performance Comparison Table:**

| Metric        | LR (4 Feat) | LR (13 Feat) | XGB (4 Feat) | XGB (13 Feat) |
| :------------ | :---------- | :----------- | :----------- | :------------ |
| **AUC-ROC**   | 0.7523      | **0.7580**   | 0.7483       | 0.7458        |
| Accuracy      | **0.8306**  | **0.8308**   | 0.7123       | 0.7134        |
| Precision 0   | **0.67**    | 0.65         | 0.35         | 0.35          |
| Recall 0      | 0.16        | 0.18         | **0.64**     | 0.63          |
| F1-Score 0    | 0.26        | 0.28         | **0.45**     | **0.45**      |
| Precision 1   | 0.8385      | 0.8409       | **0.9001**   | 0.8983        |
| Recall 1      | **0.9813**  | 0.9775       | 0.7280       | 0.7314        |
| F1-Score 1    | **0.9043**  | 0.9041       | 0.8049       | 0.8063        |
*(Note: Random Forest results were generally worse than LR or XGB in terms of AUC and are omitted for brevity)*

**Key Observations for XGBoost (Updated Features):**

1.  **Feature Impact:** Comparing XGB (13 Feat) vs. XGB (4 Feat), the additional features resulted in **no significant improvement**. AUC actually decreased slightly (0.7483 -> 0.7458), and the performance balance between Class 0 and Class 1 remained almost identical.
2.  **XGB vs LR (Updated Features):** Comparing XGB (13 Feat) vs LR (13 Feat), the fundamental trade-off remains:
    *   LR (13 Feat) has slightly better AUC (0.7580 vs 0.7458) and much higher Accuracy (due to better Class 1 recall).
    *   XGB (13 Feat) remains far superior at identifying the minority class (Recall 0: 0.63 vs 0.18; F1 0: 0.45 vs 0.28).
3.  **Updated Feature Importance (XGBoost):**
    *   `user_reorder_ratio` and `user_avg_basket_size` still dominate.
    *   The new features generally show low importance scores in this XGBoost model. `user_avg_unique_prods_per_order` surprisingly received 0 importance, suggesting it might be redundant given other features (like average basket size) or simply didn't provide useful split points for the trees XGBoost built.

**Overall Conclusion & Next Steps Discussion**

This iteration of feature engineering yielded only marginal improvements for Logistic Regression and virtually none for XGBoost (with default settings). This strengthens the hypothesis that achieving substantially better performance might require either:

*   **A) More Sophisticated Feature Engineering:** Creating features that capture even more nuanced user behavior, possibly interaction terms, or time-decayed patterns.
*   **B) Hyperparameter Tuning:** Optimizing the settings of the models (especially XGBoost) could potentially unlock better performance from the *current* 13-feature set. Default settings are rarely optimal.
*   **C) Different Model Architectures:** Exploring fundamentally different models (like Neural Networks) could be considered, but usually only after exhausting options A and B.

This concludes the modeling section based on the features engineered so far. We have established baselines, explored improvements via added features, evaluated standard advanced models, and identified the next logical steps for optimization.
            """)

st.markdown("""
# Phase 6: Summary of Modeling & Refined Next Steps

## Modeling Summary

We systematically built and evaluated models using progressively richer feature sets:

1.  **Baseline (4 Features):** Logistic Regression established an AUC of 0.7523 but had very poor recall (0.16) for the minority class (Class 0 - No New Product).
2.  **Advanced Models (4 Features):** Random Forest underperformed the baseline AUC. XGBoost matched the baseline AUC but dramatically improved minority recall (0.64) at the cost of overall accuracy and majority recall.
3.  **Richer Features (13 Features):** Adding 9 more features yielded slight improvements for Logistic Regression (AUC up to 0.7580, Recall 0 up to 0.18) but negligible improvement for XGBoost (AUC 0.7458, Recall 0 at 0.63).
4.  **Persistent Trade-off:** A clear choice remains between the model with the best overall AUC/accuracy (Logistic Regression on 13 features) and the model best at identifying the crucial minority class (XGBoost on either feature set).

## Recommended Next Steps

Given that the richer feature set provided limited gains with default model settings, the most logical paths forward are:

1.  **Hyperparameter Tuning (Highest Priority):**
    *   **Focus:** Primarily on XGBoost, as it showed the most potential for handling the minority class effectively. Tuning parameters like `n_estimators`, `max_depth`, `learning_rate`, `gamma`, `subsample`, `colsample_bytree` could significantly improve its AUC and potentially find an even better balance.
    *   **Method:** Use `GridSearchCV` or `RandomizedSearchCV` with cross-validation on the training set (`X_train_upd`, `y_train_upd`) using AUC or F1-score for Class 0 as the scoring metric.
    *   **Also Consider:** Tuning the `C` parameter for Logistic Regression could also yield slight improvements.

2.  **Advanced Feature Engineering (If Tuning is Insufficient):**
    *   If optimized models still don't meet performance goals, revisit feature engineering.
    *   **Ideas:** Features capturing user tenure, rolling averages of behavior, time-decayed features (giving more weight to recent orders), product category embedding features, interaction terms between key predictors.

3.  **Threshold Adjustment:** For the chosen *best* model (after tuning), analyze its probability outputs (`predict_proba`) and potentially select a probability threshold different from the default 0.5 to optimize for a specific balance of Precision and Recall based on business needs (e.g., using a Precision-Recall curve).

For the scope of demonstrating the ML process, **Hyperparameter Tuning** is the most standard and important next step after initial model evaluation.
            """)

st.markdown("""
# Phase 7: Hyperparameter Tuning (Optimizing XGBoost)

Our previous steps showed that different models offer different strengths, and adding features provided only modest gains with default settings. A crucial step in improving model performance is **Hyperparameter Tuning**.

**What are Hyperparameters?**
These are settings for the machine learning algorithm itself that are *not* learned from the data during training. Instead, they are set *before* training starts. Examples for XGBoost include:
*   `n_estimators`: The number of decision trees to build sequentially.
*   `max_depth`: The maximum depth allowed for each individual tree.
*   `learning_rate`: How much each new tree contributes to the overall prediction (controls the step size).
*   `subsample`: The fraction of training data samples used to build each tree.
*   `colsample_bytree`: The fraction of features considered when building each tree split.
*   `gamma`: Minimum loss reduction required to make a further partition on a leaf node (acts as regularization).

**Why Tune Them?**
Default hyperparameter values are often not optimal for a specific dataset. Finding the right combination can significantly impact model performance, potentially leading to better accuracy, AUC, or a better balance between precision and recall.

**Our Approach: Randomized Search**
We will use `RandomizedSearchCV` from Scikit-learn. Instead of trying every single possible combination like `GridSearchCV` (which can be very slow), `RandomizedSearchCV` samples a fixed number of parameter combinations from specified distributions. It's often more efficient for exploring a wide range of possibilities.

*   **Model:** We will focus on tuning the **XGBoost** model, as it showed the best Recall for the minority class.
*   **Data:** Tuning will be performed using **cross-validation** *only on the training set* (`X_train_upd`, `y_train_upd`) to prevent leakage of information from the test set.
*   **Goal:** Find the hyperparameter combination that maximizes a chosen metric (e.g., AUC-ROC, which reflects overall discriminatory power) during cross-validation.
*   **Final Evaluation:** Once the best parameters are found, we will train a *final* XGBoost model using these parameters on the *entire* training set and evaluate it once on the *held-out test set* (`X_test_upd`, `y_test_upd`).
            """)

st.code("""
Best Hyperparameters found:
{'colsample_bytree': 0.8721, 'gamma': 0.2252, 'learning_rate': 0.0139, 'max_depth': 3, 'n_estimators': 415, 'scale_pos_weight': 0.2261, 'subsample': 0.8253} 
        """)

st.subheader("Tuned XGBoost Features Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7605", delta=f"{(0.7605-0.7483):.4f} vs Default XGB")
col2.metric("Precision (Class 0)", "0.36", delta=f"{(0.36-0.35):.4f} vs Default XGB")
col3.metric("Recall (Class 0)", "0.66", delta=f"{(0.66 - 0.64):.4f} vs Default XGB")
col4.metric("F1-Score", "0.46", delta=f"{(.46-.45):.4f} vs Default XGB")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("XGBoost with Tuned Hyperparameters Features Confusion Matrix")
        st.image('images/XGB3_Confusion_Matrix.png', caption='Confusion Matrix for our updated Logistic Regression')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/LR2_Confusion_Matrix.png' exists.")

st.write("")

st.markdown("""
### Analysis: Tuned XGBoost Results

We performed `RandomizedSearchCV` to find better hyperparameters for XGBoost, optimizing for AUC during cross-validation on the training set. We then trained a final model with these best parameters and evaluated it on the held-out test set.

**Key Findings from Tuning:**

1.  **Best CV Score:** The best average AUC score achieved during the 3-fold cross-validation search was **0.7617**. This is higher than the default XGBoost's test set AUC (0.7458) and also slightly higher than the best score achieved by the tuned Logistic Regression (0.7580), suggesting tuning was beneficial during the search phase.
2.  **Best Hyperparameters:** The search identified specific values (e.g., `max_depth=3`, `n_estimators=415`, a low `learning_rate` around 0.014, specific `subsample` and `colsample_bytree` rates) as optimal for AUC on the training folds. Notice the `scale_pos_weight` remained fixed as intended.

**Evaluation on Test Set (Tuned XGBoost):**

Let's compare the *tuned* XGBoost model's performance on the *test set* against the previous best models:

| Metric        | LR (13 Feat) | XGB (13 Feat, Default) | XGB (13 Feat, Tuned) | Change (Tuned XGB vs Best Previous AUC) |
| :------------ | :----------- | :--------------------- | :------------------- | :-------------------------------------- |
| **AUC-ROC**   | **0.7580**   | 0.7458                 | **0.7605**           | **Improved (+0.0025 vs LR)**            |
| Accuracy      | **0.8308**   | 0.7134                 | 0.7169               | Worse (vs LR), Similar (vs XGB Def)     |
| Precision 0   | **0.65**     | 0.35                   | 0.36                 | Worse (vs LR), Similar (vs XGB Def)     |
| **Recall 0**    | 0.18         | 0.63                   | **0.66**             | **Improved (+0.03 vs XGB Def)**         |
| **F1-Score 0**  | 0.28         | 0.45                   | **0.46**             | **Improved (+0.01 vs XGB Def)**         |
| Precision 1   | 0.8409       | 0.8983                 | **0.9051**           | **Improved (+0.0068 vs XGB Def)**       |
| Recall 1      | **0.9775**   | 0.7314                 | 0.7293               | Worse (vs LR), Similar (vs XGB Def)     |
| F1-Score 1    | **0.9041**   | 0.8063                 | 0.8078               | Worse (vs LR), Similar (vs XGB Def)     |

**Analysis of Tuned XGBoost:**

1.  **AUC Improvement Achieved:** The tuned XGBoost model achieved an AUC of **0.7605** on the test set. This is the **highest AUC score** we've seen so far, surpassing the tuned Logistic Regression (0.7580). Hyperparameter tuning successfully improved the model's overall ability to distinguish between classes.
2.  **Minority Class Boost:** Tuning also led to improvements in identifying the minority class (Class 0). Recall increased further to **0.66** (from 0.63 with default XGBoost), and the F1-score nudged up to **0.46**. This tuned model is now identifying two-thirds of the users who don't buy new products.
3.  **Trade-off Persists:** The fundamental trade-off remains. Achieving this high minority recall comes at the expense of lower overall accuracy and lower recall for the majority class compared to Logistic Regression. The precision for Class 0 also remains relatively low (0.36).
4.  **Final "Best" Model:** Based on achieving the highest AUC and the best performance metrics for the challenging minority class (Recall 0, F1 0), the **tuned XGBoost model represents the best overall result** from our experiments so far.

**Conclusion for Modeling Phase:**

Through iterative feature engineering and model tuning, we arrived at an optimized XGBoost model that provides the best balance found between overall discriminatory power (AUC) and the crucial ability to identify the minority class (users not buying new products), surpassing our initial baseline models. While perfect prediction isn't achieved, the model provides significantly better-than-random insights.
            """)

st.markdown("""
# Phase 8: Final Summary & Future Directions

## Project Recap & Achievements

We successfully executed an end-to-end machine learning workflow to predict if an Instacart user would purchase a new product in their next order:
1.  **Goal & Data:** Defined the prediction task using Instacart data and established 'new product purchase' as a proxy for marketing success/user exploration.
2.  **Target Variable:** Calculated the binary target `new_product_purchased`.
3.  **Feature Engineering:** Created an initial set of 4 user-level features and later expanded to 13 features capturing timing, diversity, and reorder patterns based on prior purchase history.
4.  **Modeling & Evaluation:**
    *   Built a Logistic Regression baseline.
    *   Tested Random Forest, observing trade-offs with class weighting.
    *   Evaluated XGBoost, noting its effectiveness (with `scale_pos_weight`) in identifying the minority class.
    *   Retrained models with expanded features, finding modest gains for LR but limited impact on XGBoost defaults.
5.  **Optimization:** Performed hyperparameter tuning on XGBoost using `RandomizedSearchCV`, resulting in the best overall performance achieved in this project (AUC = 0.7605, Class 0 Recall = 0.66).

## Key Findings & Model Choice

*   **Best Model:** The **tuned XGBoost model** emerged as the strongest performer, offering the best balance found between overall discriminatory power (highest AUC achieved) and significantly improved identification of the minority class (highest Recall and F1 for Class 0).
*   **Feature Importance:** Across models, `user_reorder_ratio` and `user_avg_basket_size` consistently proved to be the most influential predictors.
*   **Challenge:** Accurately predicting new product purchases, especially identifying the ~18% of users who *don't* buy new items, remains challenging even with optimization, suggesting potential limits to the current feature set or requiring more advanced techniques.
*   **Trade-offs:** A clear trade-off exists between models optimized for overall AUC/accuracy (like LR) and those optimized for minority class recall (like tuned XGBoost). The choice depends on the specific business objective.

## Future Directions & "Part B"

While this concludes the core demonstration of building a user-level prediction model, several exciting avenues exist for future work:

**Refining the Current Model:**
1.  **Advanced Feature Engineering:** Develop more sophisticated features (time-decayed, category-based, lag features, interaction terms) to potentially break through the current performance ceiling.
2.  **More Tuning/Other Models:** Conduct more extensive hyperparameter searches or explore other algorithms (e.g., LightGBM, Neural Networks) if features alone aren't sufficient.
3.  **Threshold Optimization:** Fine-tune the probability threshold of the final model based on specific precision/recall goals for deployment.
4.  **Error Analysis:** Deeply analyze the samples the best model misclassifies to guide further improvements.

**"Part B" - Product-Level Prediction:**
5.  **Reformulate the Problem:** Shift from predicting if a *user* buys *any* new product to predicting the probability that a *specific user* buys a *specific new product*.
    *   **Requires:** User-product level training data, new features describing product characteristics and user-product interactions, and different model evaluation strategies.
    *   **Benefit:** Enables highly targeted recommendations or promotions for individual products a user hasn't tried before.

This project provides a solid foundation and a functional model for predicting general user exploration, while also outlining clear paths for future enhancement and expansion into more granular, product-specific predictions.
            """)

st.markdown("""
# Phase 9 (Revised): Refining the Best Model - Threshold Adjustment for Class 0

Our tuned XGBoost model showed the best potential for identifying the minority class (Class 0 - No New Product), achieving a Recall of 0.66 with the default 0.5 probability threshold. However, this default threshold might not yield the optimal balance of Precision and Recall *specifically for Class 0*.

**Why Adjust the Threshold for Class 0?**
The default 0.5 threshold is arbitrary. By adjusting the probability threshold required to classify an instance as Class 1 (New), we inversely affect the classification of Class 0.
*   **Higher Threshold (for predicting Class 1):** Fewer instances predicted as Class 1 -> More instances predicted as Class 0 -> Higher Recall 0, Lower Precision 0.
*   **Lower Threshold (for predicting Class 1):** More instances predicted as Class 1 -> Fewer instances predicted as Class 0 -> Lower Recall 0, Higher Precision 0.

Our goal here is to find the threshold that gives the **best F1-Score for Class 0**, effectively finding the operating point that best balances Precision and Recall *for identifying users who likely won't buy new products, say if we have a set budget for advertising and want to target users we know are more likely to buy*.

We will now:
1. Iterate through possible thresholds using the predicted probabilities from the tuned XGBoost model.
2. Calculate Precision, Recall, and F1-score *specifically for Class 0* at each threshold.
3. Identify the threshold that maximizes the F1-score for Class 0.
4. Evaluate the model's performance using this Class 0 optimized threshold.
            """)

st.subheader("XGBoost with Optimal Threshold Features Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7605", delta=f"{(0.7605-0.7605):.4f} vs Tuned XGB")
col2.metric("Precision (Class 0)", "0.38", delta=f"{(0.38-0.36):.4f} vs Tuned XGB")
col3.metric("Recall (Class 0)", "0.60", delta=f"{(0.60 - 0.66):.4f} vs Tuned XGB")
col4.metric("F1-Score", "0.47", delta=f"{(.47-.46):.4f} vs Tuned XGB")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 5, 1])
with col3:
    try:
        st.subheader("Class 0 Metric vs. Prediction Threshold")
        st.image('images/class0metrics_vs_predictionthreshold.png', caption='chart showing the trendlines for the model metrics, you can see where the optimal threshold is found')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/class0metrics_vs_predictionthreshold.png' exists.")

st.write("")
with col4:
    try:
        st.subheader("Tuned XGBoost Confusion Matrix")
        st.image('images/FinalXGB_Confusion_Matrix.png', caption='Confusion Matrix of the Tuned XGBoost model')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/FinalXGB_Confusion_Matrix.png' exists.")

st.write("")

st.markdown("""
### Analysis: Threshold Adjustment Optimized for Class 0

We recalculated performance across different probability thresholds, this time specifically aiming to maximize the **F1-Score for the minority Class 0 (No New Product)** using our tuned XGBoost model.

**Key Findings:**

1.  **Optimal Threshold for Class 0 F1 (0.4540):** The threshold that best balances Precision and Recall *for Class 0* was found to be 0.4540. This is significantly higher than the threshold that optimized F1 for Class 1 (0.1734) and slightly lower than the default 0.5.
2.  **Class 0 Performance at Optimal Threshold:**
    *   **F1-Score (Class 0):** Achieved the maximum value found of **0.47 (0.4676)**. This is slightly better than the F1-score achieved using the default 0.5 threshold (0.46 from the previous report) and much better than the F1 achieved when optimizing for Class 1 (0.27).
    *   **Recall (Class 0):** Reached **0.60**. This is lower than the recall achieved with the default 0.5 threshold (0.66) but vastly better than optimizing for Class 1 (0.16). We are still identifying a good portion (60%) of users who don't buy new products.
    *   **Precision (Class 0):** Improved to **0.38** compared to the default threshold (0.36). When the model predicts Class 0 using this threshold, it's correct 38% of the time.
3.  **Impact on Class 1 Performance:** As expected, optimizing for Class 0 slightly impacts Class 1:
    *   Recall (Class 1) decreased to 0.7796 (compared to 0.9826 when optimizing for Class 1 F1).
    *   Precision (Class 1) remained high at 0.8966.
    *   F1-Score (Class 1) decreased to 0.8340.
4.  **Overall Metrics:**
    *   Accuracy (0.7469) is higher than with the default 0.5 threshold (0.7169) but lower than when optimizing for Class 1 F1 (0.8318).
    *   AUC remains unchanged (0.7605), as it's threshold-independent.

**Conclusion:**

Optimizing the threshold specifically to maximize the F1-score for Class 0 yields a threshold of **0.4540**. This operating point provides the best balance found *between Precision and Recall for the minority class*, achieving an F1-score of 0.47 and identifying 60% of these users (Recall=0.60).

This contrasts with:
*   The **default 0.5 threshold**, which gave slightly higher Recall (0.66) but lower Precision (0.36) and F1 (0.46) for Class 0.
*   The **Class 1 F1-optimized threshold (0.1734)**, which completely sacrificed Class 0 Recall (0.16) for maximizing Class 1 performance.

**Choosing the Right Threshold:** The choice between the default 0.5 threshold and this optimized 0.4540 threshold depends on whether the priority for Class 0 is slightly higher *recall* (finding more, even with more false positives - use default 0.5) or the best *balance* of precision and recall (use 0.4540). Both are significant improvements over the initial Logistic Regression model for identifying this group. It really depends on the business application! Which type of error is more costly? Missing a user who won't buy new (FN for Class 0)? Or wrongly predicting a user won't buy new when they will (FP for Class 0)?

This completes our threshold analysis, showing how we can fine-tune the model's output for different objectives.
            """)

st.markdown("""
# Phase 10: Error Analysis

We have trained, evaluated, and optimized our best model (tuned XGBoost) and its decision threshold. Now, let's try to understand *where* it's still making mistakes. Analyzing these errors can provide valuable insights for potential future improvements (like better feature engineering).

**Goal:** Identify patterns in the users that the model misclassifies.

**Types of Errors (Focusing on the Class 0 Optimal Threshold ≈ 0.45):**
We'll examine the two main types of errors based on the confusion matrix generated using the threshold optimized for Class 0 F1:

1.  **False Positives (FP):**
    *   **What they are:** Users the model predicted would buy a new product (Predicted 1), but they actually *did not* (Actual 0).
    *   **Why analyze:** These might represent users who look like explorers based on our features but stick to familiar items. Understanding them might reveal missing features or limitations in our current ones.
2.  **False Negatives (FN):**
    *   **What they are:** Users the model predicted would *not* buy a new product (Predicted 0), but they actually *did* (Actual 1).
    *   **Why analyze:** These are users whose exploration behavior wasn't captured by the model. Why did the model think they wouldn't buy new things?

**Approach:**
1. Identify the `user_id`s corresponding to FP and FN on the test set.
2. Compare the distribution of features (`X_test_upd`) for these error groups against the correctly classified groups (True Positives and True Negatives).
3. Look for significant differences in feature values (e.g., do False Positives have unusually high basket sizes but also high reorder ratios?).
            """)

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("Comparing False Positives vs True Negatives for Each Predictor")
        st.image('images/FP_vs_TN.png', caption='Plotting distributions for key features (FP vs TN and FN vs TP)')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/LR2_Confusion_Matrix.png' exists.")

st.write("")

st.markdown("""
### Analysis: Understanding Model Errors (Tuned XGBoost, Threshold ≈ 0.45)

We analyzed the characteristics of the users that our tuned XGBoost model (using the threshold optimized for Class 0 F1-score) misclassified on the test set.

**Summary Statistics Comparison:**

| Feature                       | False Positive (Predicted 1, Actual 0) | True Negative (Predicted 0, Actual 0) | False Negative (Predicted 0, Actual 1) | True Positive (Predicted 1, Actual 1) | Notes                                                                                                  |
| :---------------------------- | :------------------------------------- | :------------------------------------ | :------------------------------------- | :------------------------------------ | :----------------------------------------------------------------------------------------------------- |
| **user_reorder_ratio** (Mean) | 0.407 (Lower)                          | **0.642 (Higher)**                    | **0.588 (Higher)**                     | 0.355 (Lower)                         | **Main Driver:** TN and FN (Actual 0 & Predicted 0) have much higher reorder ratios than TP and FP.    |
| **user_avg_basket_size** (Mean) | **10.41 (Higher)**                     | 5.59 (Lower)                          | 6.35 (Lower)                           | **11.70 (Higher)**                    | TP and FP (Predicted 1) have much larger average baskets than TN and FN (Predicted 0).                 |
| user_avg_days_since_prior (Mean)| 15.5 (Longer)                          | 12.7 (Shorter)                        | 13.1 (Shorter)                         | 16.1 (Longer)                         | Predicted 1 users tend to wait longer between orders. Predicted 0 users shop slightly more frequently. |
| user_total_orders (Mean)        | 13.6 (Lower)                           | **26.5 (Higher)**                     | **24.6 (Higher)**                      | 11.4 (Lower)                          | Predicted 0 users (TN, FN) tend to have longer order histories than Predicted 1 users (TP, FP).        |
| user_total_departments (Mean) | **11.4 (Higher)**                      | 8.4 (Lower)                           | 9.3 (Lower)                            | **11.7 (Higher)**                     | Predicted 1 users tend to buy from more departments (higher diversity).                                |
| user_total_items_purchased (Mean)| 155.8                                  | 185.0                                 | **197.1**                              | 139.7                                 | FN users have bought the most items overall, suggesting maybe their "new" item was an anomaly?         |

**Interpreting the Mistakes:**

1.  **False Positives (Predicted 1, Actual 0):**
    *   *Why did the model predict they'd buy new?* Compared to True Negatives (who also didn't buy new), False Positives tend to have **significantly lower `user_reorder_ratio`**, **larger average basket sizes**, wait slightly longer between orders, have shorter order histories, and buy from more departments.
    *   *In short:* They *look* more like typical explorers (lower reorder ratio, higher diversity, larger baskets) based on averages, but in their *specific* 'train' order, they happened to only buy familiar items. The model was "fooled" by their generally exploratory profile.

2.  **False Negatives (Predicted 0, Actual 1):**
    *   *Why did the model predict they *wouldn't* buy new?* Compared to True Positives (who also bought new), False Negatives tend to have **significantly higher `user_reorder_ratio`**, **smaller average basket sizes**, shop slightly more frequently, and have much longer order histories.
    *   *In short:* They *look* more like habitual shoppers who mostly reorder (high reorder ratio, smaller baskets, longer history), but in their *specific* 'train' order, they surprisingly bought something new. The model missed this deviation from their typical pattern.

**Insights & Potential Improvements:**

*   **Reorder Ratio is Key but Imperfect:** While `user_reorder_ratio` is the strongest signal, relying too heavily on this *average* causes errors. Users can deviate from their average behavior.
*   **Basket Size Matters:** Users predicted to buy new things (TP, FP) consistently have larger average baskets.
*   **History Length:** Users predicted *not* to buy new things (TN, FN) tend to be longer-term customers. Perhaps loyalty or habit strength isn't fully captured.
*   **Need for Recency/Lag Features?** The errors suggest average historical behavior isn't enough. Features describing the *most recent* orders (e.g., size of the very last basket, time since *that* order, was the *last* order unusually large/small?) might help capture deviations from the average that predict behavior in the *next* order.
*   **Category Exploration:** While total departments/aisles helps, maybe features about *which specific* categories are explored vs. consistently repurchased would add value.

**Conclusion for Error Analysis:**

The model primarily distinguishes users based on their historical reorder tendency and average basket size. Errors occur when users deviate from their typical profile in the target 'train' order. False Positives look like explorers but didn't explore *this time*, while False Negatives look like non-explorers but *did* explore *this time*. This points towards adding features related to order recency and potentially more granular category behavior as promising avenues for future improvement.
            """)

st.markdown("""
# Phase 3c: Advanced Feature Engineering - Recency

Our error analysis suggested that average historical behavior might not be enough. Errors often occurred when users deviated from their averages in the target 'train' order. Features describing the user's *most recent* behavior before the 'train' order might capture important signals about their current state or intent.

We will now engineer features specifically related to the user's **last prior order**:

1.  **`days_since_last_order`**: The exact number of days that passed between the user's *last prior order* and their current 'train' order. (This directly uses the `days_since_prior_order` value associated with the 'train' order itself).
2.  **`last_order_basket_size`**: The total number of items the user purchased in their *immediately preceding* prior order.

These features provide context about the specific transition into the 'train' order we are trying to predict.
            """)

st.code("""
   days_since_last_order  last_order_basket_size  
0                   14.0                       9  
1                   30.0                      16  
2                    6.0                      12  
3                    6.0                      12  
4                   10.0                      13  
        """)

st.markdown("""
# Phase 4 (Repeated Again): Preparing Final Data for Modeling

We have now added recency features, bringing our total feature count to 15 (plus `user_id`).

We need to repeat the data preparation steps one more time using this **final feature set**:

1.  **Merging:** Combine the latest `features_df` (with 15 features) and the `final_target_df`.
2.  **Splitting:** Divide this dataset into new Training and Testing sets (`X_train`, `X_test`, `y_train`, `y_test`), using the same `test_size` (20%), `random_state` (42), and `stratify=y` settings for consistency.

This provides the final inputs for our model evaluation phase based on all engineered features.
            """)

st.markdown("""
# Phase 5 (Final Run): Evaluating Models with Final Feature Set (15 Features)

We have now incorporated recency features (`days_since_last_order`, `last_order_basket_size`) into our feature set, bringing the total to 15 descriptive variables per user.

This is our final feature engineering step for this iteration. We will now retrain and re-evaluate our two most promising models on this complete feature set:

1.  **Logistic Regression:** To see if the linear model benefits from the recency information.
2.  **Tuned XGBoost:** To see if the optimized non-linear model can leverage these new features for further performance gains, particularly in AUC or minority class recall.

We will compare the results to the previous runs (using 4 and 13 features) to assess the final impact of our feature engineering efforts.
            """)

st.subheader("Final XGBoost Features Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.7803", delta=f"{(0.7803-0.7605):.4f} vs Optimal Threshold XGB")
col2.metric("Precision (Class 0)", "0.36", delta=f"{(0.36-0.38):.4f} vs Optimal Threshold XGB")
col3.metric("Recall (Class 0)", "0.70", delta=f"{(0.70 - 0.60):.4f} vs Optimal Threshold XGB")
col4.metric("F1-Score", "0.48", delta=f"{(.48-.47):.4f} vs Optimal Threshold XGB")

col1, col2, col3, col4, col5 = st.columns([1, 1, 5, 1, 1])
with col3:
    try:
        st.subheader("Final XGBoost Feature Importances")
        st.image('images/FinalXGB_Feature_Importances.png', caption='Bar chart of Final XGBoost Feature Importances')
    except FileNotFoundError:
        st.error("Plot image not found. Ensure 'images/FinalXGB_Feature_Importances.png' exists.")

st.write("")




st.markdown("""
# Final Model Evaluation & Project Conclusion

We have completed the final round of model evaluation using the full set of 15 engineered features, including recency metrics.

**Final Performance Comparison Table:**

| Metric        | LR (4 Feat) | LR (13 Feat) | LR (15 Feat) | XGB (4 Feat) | XGB (13 Feat, Tuned) | XGB (15 Feat, Tuned) |
| :------------ | :---------- | :----------- | :----------- | :----------- | :------------------- | :------------------- |
| **AUC-ROC**   | 0.7523      | 0.7580       | 0.7676       | 0.7483       | 0.7605               | **0.7803**           |
| Accuracy      | 0.8306      | 0.8308       | **0.8314**   | 0.7123       | 0.7169               | 0.7182               |
| Precision 0   | **0.67**    | 0.65         | 0.64         | 0.35         | 0.36                 | 0.36                 |
| Recall 0      | 0.16        | 0.18         | 0.19         | 0.64         | 0.66                 | **0.70**             |
| F1-Score 0    | 0.26        | 0.28         | 0.30         | 0.45         | 0.46                 | **0.48**             |
| Precision 1   | 0.8385      | 0.8409       | 0.8425       | 0.9001       | 0.9051               | **0.9131**           |
| Recall 1      | **0.9813**  | 0.9775       | 0.9756       | 0.7280       | 0.7293               | 0.7233               |
| F1-Score 1    | **0.9043**  | 0.9041       | **0.9042**   | 0.8049       | 0.8078               | 0.8072               |

**Analysis of Final Results:**

1.  **Feature Engineering Impact:** Adding the final two recency features (`days_since_last_order`, `last_order_basket_size`) provided another boost to performance for *both* models.
    *   **Logistic Regression:** AUC improved further to **0.7676** (from 0.7580). Recall for Class 0 also slightly increased to **0.19**.
    *   **Tuned XGBoost:** AUC saw a more significant jump to **0.7803** (from 0.7605). Crucially, **Recall for Class 0 reached 0.70** (up from 0.66), and the F1-score for Class 0 improved to **0.48**. Precision for Class 1 also hit a high of **0.9131**.
2.  **Final Model Champion:** The **Tuned XGBoost model trained on the final 15 features** is clearly the best-performing model developed in this project. It achieves the highest AUC (0.7803) and provides by far the best capability for identifying the minority class (Recall 0 = 0.70, F1 0 = 0.48), while maintaining very high precision for the majority class (Precision 1 = 0.91).
3.  **New Feature Importance (XGBoost):** The final XGBoost feature importance plot shows that the new recency feature `last_order_basket_size` became the *most important* feature, surpassing even `user_reorder_ratio` and `user_avg_basket_size`. `days_since_last_order` also showed reasonable importance. This confirms that information about the *immediately preceding* order is highly valuable for predicting behavior in the *next* order.

**Overall Project Conclusion:**

Through a structured process involving data exploration, clear target definition, iterative feature engineering (culminating in 15 user-level features including recency), baseline modeling, evaluation of advanced algorithms, and hyperparameter tuning, we successfully built an XGBoost model capable of predicting whether an Instacart user will purchase a new product in their next order with reasonable accuracy (AUC = 0.7803).

The final model demonstrates a strong ability to identify the challenging minority group (users *not* buying new products, Recall=0.70) while maintaining high precision when predicting the majority group (Precision=0.91). Key drivers identified include the user's historical reorder ratio, average basket size, and particularly the size of their last order.

This project serves as a practical demonstration of the machine learning workflow, highlighting the importance of feature engineering, model selection trade-offs (especially with imbalanced data), and optimization techniques like hyperparameter tuning and threshold analysis to arrive at a useful predictive model. Potential future work could involve even more advanced features or exploring the "Part B" task of predicting specific product purchases.
            """)







