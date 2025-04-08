import streamlit as st
import pandas as pd
# from PIL import Image # Import if/when you use st.image later

st.set_page_config(layout="wide") # Use wide layout for better space

st.title("Instacart Market Basket Analysis")

# --- Add Buttons Below Title ---

# Define custom CSS for button colors using Markdown
# We'll wrap buttons in divs with specific IDs to target them
st.markdown("""
<style>
/* Style for the container div to center buttons if necessary */
.button-container {
    display: flex;
    justify-content: center;
    gap: 20px; /* Adds space between buttons */
    margin-top: 20px;
    margin-bottom: 30px;
}

/* Target the specific divs and style the Streamlit button inside them */
#button-a .stButton button {
    background-color: #4CAF50; /* Green */
    color: white;
    border-radius: 5px;
    padding: 10px 24px;
    border: none;
    font-weight: bold;
}

#button-b .stButton button {
    background-color: #FF9800; /* Orange */
    color: white;
    border-radius: 5px;
    padding: 10px 24px;
    border: none;
    font-weight: bold;
}

/* Optional: Hover effects */
#button-a .stButton button:hover {
    background-color: #45a049;
}
#button-b .stButton button:hover {
    background-color: #fb8c00;
}

</style>
""", unsafe_allow_html=True)

# Use columns to help center the buttons horizontally
# Adjust the numbers in the list for different spacing (left_spacer, content, right_spacer)
# Using equal spacers on left/right pushes content towards center.
col1, col2, col3 = st.columns([1, 2, 1]) # Experiment with these ratios, e.g., [1, 1, 1] or [2, 3, 2]

with col2: # Place buttons in the central column
    # Use another set of columns *inside* the central one for side-by-side layout
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        # Wrap the button in a div with ID "button-a"
        st.markdown('<div id="button-a">', unsafe_allow_html=True)
        # Add the Streamlit button - it won't do anything yet
        part_a_clicked = st.button("Part A: Project Narrative", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b_col2:
        # Wrap the button in a div with ID "button-b"
        st.markdown('<div id="button-b">', unsafe_allow_html=True)
         # Make this button disabled for now, since Part B isn't built
        part_b_clicked = st.button("Part B: Product Predictor (Soon!)", disabled=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Add a horizontal rule after buttons for separation
st.markdown("---")










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

st.subheader("Logistic Regression Evaluation Metrics")
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























# --- Phase 5 (Example: Showing Final Model Results) ---
st.header("Final Model Evaluation (Tuned XGBoost - 15 Features)")

st.subheader("Performance Metrics")
# You can use st.metric for key results
col1, col2, col3 = st.columns(3)
col1.metric("Final AUC", "0.7803", delta="0.0198 vs Default XGB") # Example delta
col2.metric("Recall (Class 0)", "0.70", delta="0.04 vs Default XGB")
col3.metric("F1 (Class 0)", "0.48", delta="0.02 vs Default XGB")

st.subheader("Classification Report")
st.code("""
# Output from Notebook:
#                   precision    recall  f1-score   support
# Class 0 (No New)       0.36      0.70      0.48      4840
#    Class 1 (New)       0.91      0.72      0.81     21402
#         accuracy                           0.72     26242
# ... etc ...
""", language=None)

st.subheader("Confusion Matrix")
try:
    st.image('images/xgb_cm_tuned_final.png', caption='Confusion Matrix for Tuned XGBoost (15 Features)')
except FileNotFoundError:
    st.error("Plot image not found. Ensure 'images/xgb_cm_tuned_final.png' exists.")

st.subheader("Feature Importances")
try:
    st.image('images/xgb_feat_imp_final.png', caption='Feature Importances for Tuned XGBoost (15 Features)')
except FileNotFoundError:
    st.error("Plot image not found. Ensure 'images/xgb_feat_imp_final.png' exists.")

# ... Continue structuring for all phases ...

st.header("Project Conclusion")
st.markdown("""
Through a structured process... we successfully built an XGBoost model...
... (rest of your final conclusion Markdown) ...
""")