
"""
RecMetrics Lite - A streamlined library for recommender system evaluation metrics.
This single file contains all core functionality from the recmetrics library,
with compatibility for modern Python package versions and minimal dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve, roc_curve, average_precision_score
from scipy import stats

# Define a color palette that doesn't depend on seaborn
RECOMMENDER_PALETTE = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]

#---------------------------------
# METRIC FUNCTIONS
#---------------------------------

def coverage(predicted, catalog):
    """
    Computes the coverage for a list of recommendations.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    catalog : list
        list of all unique items in training data
    Returns
    -------
    float
        coverage
    """
    predicted_flattened = [p for sublist in predicted for p in sublist]
    return len(set(predicted_flattened)) / len(catalog)

def catalog_coverage(predicted, catalog, k):
    """
    Computes the catalog coverage for k lists of recommendations.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    catalog : list
        list of all unique items in training data
    k : integer
        number of recommendations per user
    Returns
    -------
    float
        coverage
    """
    predicted_flat = [p for sublist in predicted for p in sublist]
    predicted_flat = predicted_flat[0:k]
    return len(set(predicted_flat)) / len(catalog)

def prediction_coverage(predicted, catalog):
    """
    Computes the prediction coverage for a list of recommendations.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    catalog : list
        list of all unique items in training data
    Returns
    -------
    float
        prediction coverage
    """
    predicted_flat = [p for sublist in predicted for p in sublist]
    return len(set(predicted_flat)) / len(catalog)

def personalization(predicted):
    """
    Personalization measures recommendation similarity across users.
    A high score indicates user recommendations are different from one another.
    A low score indicates user recommendations are very similar.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    Returns
    -------
    float
        personalization
    """
    def _dissimilarity(list1, list2):
        """Compute dissimilarity between two lists"""
        set1 = set(list1)
        set2 = set(list2)
        num_common = len(set1.intersection(set2))
        return 1.0 - (num_common / (len(set1) + len(set2) - num_common))

    # Compute pairwise dissimilarity between all user prediction lists
    n_users = len(predicted)
    total = 0.0
    count = 0
    for i in range(n_users):
        for j in range(i+1, n_users):
            total += _dissimilarity(predicted[i], predicted[j])
            count += 1
    return total / count if count > 0 else 0.0

def intra_list_similarity(predicted, feature_df):
    """
    Intra-list similarity measures diversity of items within each recommendation list.
    Parameters
    ----------
    predicted : list
        List of recommendation lists
    feature_df : pandas dataframe
        A dataframe with items as rows and features as columns
    Returns
    -------
    float
        average similarity score
    """
    def _item_similarity(item_i, item_j, feature_df):
        """Compute cosine similarity between two items based on their features"""
        i_features = feature_df.loc[item_i].values
        j_features = feature_df.loc[item_j].values

        # Calculate cosine similarity
        dot_product = np.dot(i_features, j_features)
        norm_i = np.linalg.norm(i_features)
        norm_j = np.linalg.norm(j_features)

        # Avoid division by zero
        if norm_i == 0 or norm_j == 0:
            return 0

        return dot_product / (norm_i * norm_j)

    # Calculate similarity for each list
    user_similarities = []
    for user_recommendations in predicted:
        n_items = len(user_recommendations)
        if n_items <= 1:
            continue  # Skip users with 0 or 1 recommendations

        # Calculate similarity between all pairs of items
        similarities = []
        for i in range(n_items):
            for j in range(i+1, n_items):
                item_i = user_recommendations[i]
                item_j = user_recommendations[j]
                if item_i in feature_df.index and item_j in feature_df.index:
                    similarity = _item_similarity(item_i, item_j, feature_df)
                    similarities.append(similarity)

        # Average similarity for this user
        if similarities:
            user_similarities.append(np.mean(similarities))

    # Return average similarity across all users
    return np.mean(user_similarities) if user_similarities else 0.0

def novelty(predicted, pop_df, u):
    """
    Novelty measures the mean popularity rank of recommended items.
    Parameters
    ----------
    predicted : list
        A list of lists, each inner list contains the recommended items for one user
    pop_df : pandas dataframe
        A DataFrame with an item_id column, and a popularity column (smaller = more popular)
    u : float
        Dampening factor
    Returns
    -------
    float
        novelty score
    """
    def _single_user_novelty(user_predictions, pop_df, u):
        """Calculate novelty for a single user's recommendations"""
        mean_pop = 0.0
        for item in user_predictions:
            if item in pop_df['item_id'].values:
                # Get item popularity rank
                rank = pop_df.loc[pop_df['item_id'] == item, 'popularity'].iloc[0]
                mean_pop += np.log(1 + rank) ** u

        return mean_pop / len(user_predictions) if user_predictions else 0.0

    # Calculate novelty for each user and average
    user_novelties = [_single_user_novelty(user_preds, pop_df, u) for user_preds in predicted]
    return np.mean(user_novelties)

def mse(y, y_pred):
    """
    Mean Squared Error metric.
    Parameters
    ----------
    y : numpy array or list
        True ratings
    y_pred : numpy array or list
        Predicted ratings
    Returns
    -------
    float
        MSE value
    """
    mse = np.mean(np.power(np.array(y) - np.array(y_pred), 2))
    return mse

def rmse(y, y_pred):
    """
    Root Mean Squared Error metric.
    Parameters
    ----------
    y : numpy array or list
        True ratings
    y_pred : numpy array or list
        Predicted ratings
    Returns
    -------
    float
        RMSE value
    """
    rmse = np.sqrt(mse(y, y_pred))
    return rmse

def make_confusion_matrix(y, y_pred, threshold=0.5):
    """
    Generate confusion matrix for recommendation binary classification.
    Parameters
    ----------
    y : numpy array or list
        True values (0 or 1)
    y_pred : numpy array or list
        Predicted probability values
    threshold : float, default=0.5
        Decision threshold
    Returns
    -------
    tuple
        tuple containing (confusion_matrix, accuracy, precision, recall, f1_score)
    """
    # Convert predictions to binary using threshold
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred_binary)

    # Extract components
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return cm, accuracy, precision, recall, f1

def recommender_precision(predicted, actual):
    """
    Computes precision for recommendations.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    actual : list
        list of lists of actual items
    Returns
    -------
    float
        precision
    """
    total_precision = 0.0
    for user_pred, user_actual in zip(predicted, actual):
        if not user_actual:
            continue

        # Find relevant items (items in both lists)
        n_relevant = len(set(user_pred).intersection(set(user_actual)))

        # Calculate precision for this user
        user_precision = n_relevant / len(user_pred) if user_pred else 0.0
        total_precision += user_precision

    return total_precision / len(predicted) if predicted else 0.0

def recommender_recall(predicted, actual):
    """
    Computes recall for recommendations.
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    actual : list
        list of lists of actual items
    Returns
    -------
    float
        recall
    """
    total_recall = 0.0
    for user_pred, user_actual in zip(predicted, actual):
        if not user_actual:
            continue

        # Find relevant items (items in both lists)
        n_relevant = len(set(user_pred).intersection(set(user_actual)))

        # Calculate recall for this user
        user_recall = n_relevant / len(user_actual) if user_actual else 0.0
        total_recall += user_recall

    return total_recall / len(predicted) if predicted else 0.0

def mark(predicted, actual, k=10):
    """
    Compute Mean Average Recall at k (MAR@k).
    Parameters
    ----------
    predicted : list
        list of lists of predicted items
    actual : list
        list of lists of actual items
    k : int, default=10
        number of recommendations to consider
    Returns
    -------
    float
        MAR@k
    """
    user_ar_scores = []

    for user_preds, user_actuals in zip(predicted, actual):
        if not user_actuals:
            continue

        # Consider only top k predictions
        user_preds_k = user_preds[:k]

        # Calculate recall at k for this user
        n_relevant = len(set(user_preds_k).intersection(set(user_actuals)))
        ar_score = n_relevant / len(user_actuals) if user_actuals else 0.0
        user_ar_scores.append(ar_score)

    # Return mean AR@k
    return np.mean(user_ar_scores) if user_ar_scores else 0.0

#---------------------------------
# PLOTTING FUNCTIONS
#---------------------------------

def long_tail_plot(df, item_id_column, interaction_type, percentage=None, x_labels=True):
    """
    Plots the long tail for a user-item interaction dataset.
    Parameters
    ----------
    df: pandas dataframe
        user-item interaction dataframe
    item_id_column: str
        column name identifying the item ids in the dataframe
    interaction_type: str
        type of user-item interactions
        i.e. 'purchases', 'ratings' 'interactions', or 'clicks'
    percentage: float, default=None
        percent of volume to consider as the head (percent as a decimal)
        (if default=None no line will be plotted)
    x_labels: bool, default=True
        if True, plot x-axis tick labels
    """
    # Calculate cumulative volumes
    volume_df = df[item_id_column].value_counts().reset_index()
    volume_df.columns = [item_id_column, "volume"]
    volume_df[item_id_column] = volume_df[item_id_column].astype(str)
    volume_df['cumulative_volume'] = volume_df['volume'].cumsum()
    volume_df['percent_of_total_volume'] = volume_df['cumulative_volume'] / volume_df['volume'].sum()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(0, len(volume_df))

    # Plot the line
    ax.plot(x, volume_df["volume"], color="black")
    plt.xticks(x)

    # Set labels
    ax.set_title('Long Tail Plot')
    ax.set_ylabel('# of ' + interaction_type)
    ax.set_xlabel(item_id_column)

    if percentage is not None:
        # Plot vertical line at the tail location
        head = volume_df[volume_df["percent_of_total_volume"] <= percentage]
        tail = volume_df[volume_df["percent_of_total_volume"] > percentage]
        items_in_head = len(head)
        items_in_tail = len(tail)
        plt.axvline(x=items_in_head, color="red", linestyle='--')

        # Fill area under plot
        head = pd.concat([head, tail.head(1)])
        x1 = head.index.values
        y1 = head['volume']
        x2 = tail.index.values
        y2 = tail['volume']
        ax.fill_between(x1, y1, color="blue", alpha=0.2)
        ax.fill_between(x2, y2, color="orange", alpha=0.2)

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'{items_in_head}: items in the head', markerfacecolor='blue', markersize=5),
            Line2D([0], [0], marker='o', color='w', label=f'{items_in_tail}: items in the tail', markerfacecolor='orange', markersize=5)
        ]
        ax.legend(handles=legend_elements, loc=1)
    else:
        # Just fill area under the curve
        ax.fill_between(x, volume_df['volume'], color="blue", alpha=0.3)

    if not x_labels:
        plt.xticks([], [])
        ax.set(xticklabels=[])
    else:
        ax.set_xticklabels(labels=volume_df[item_id_column], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def coverage_plot(coverage_scores, model_names):
    """
    Plots the coverage for a set of models to compare.
    Parameters
    ----------
    coverage_scores: list
        list of coverage scores in same order as model_names
    model_names: list
        list of model names in same order as coverage_scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with different colors
    bars = ax.bar(model_names, coverage_scores, color=RECOMMENDER_PALETTE[:len(model_names)])

    # Set labels
    ax.set_title('Catalog Coverage in %')
    ax.set_ylabel('coverage')

    plt.tight_layout()
    plt.show()

def personalization_plot(personalization_scores, model_names):
    """
    Plots the personalization for a set of models to compare.
    Parameters
    ----------
    personalization_scores: list
        list of personalization scores in same order as model_names
    model_names: list
        list of model names in same order as coverage_scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with different colors
    bars = ax.bar(model_names, personalization_scores, color=RECOMMENDER_PALETTE[:len(model_names)])

    # Set labels
    ax.set_title("Personalization in %")
    ax.set_ylabel("personalization")

    plt.tight_layout()
    plt.show()

def intra_list_similarity_plot(intra_list_similarity_scores, model_names):
    """
    Plots the intra-list similarity for a set of models to compare.
    Parameters
    ----------
    intra_list_similarity_scores: list
        list of intra-list similarity scores in same order as model_names
    model_names: list
        list of model names in same order as coverage_scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with different colors
    bars = ax.bar(model_names, intra_list_similarity_scores, color=RECOMMENDER_PALETTE[:len(model_names)])

    # Set labels
    ax.set_title("Similarity in %")
    ax.set_ylabel("similarity in %")

    plt.tight_layout()
    plt.show()

def mark_plot(mark_scores, model_names, k_range):
    """
    Plots the mean average recall at k for a set of models to compare.
    Parameters
    ----------
    mark_scores: list of lists
        list of list of mar@k scores over k. This list is in same order as model_names
    model_names: list
        list of model names in same order as coverage_scores
    k_range: list
        list or array identifying all k values in order
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert data to dataframe for plotting
    mark_df = pd.DataFrame(np.column_stack(mark_scores), k_range, columns=model_names)

    # Plot lines for each model
    for i, model in enumerate(model_names):
        ax.plot(k_range, mark_df[model], linewidth=3,
                label=model, color=RECOMMENDER_PALETTE[i % len(RECOMMENDER_PALETTE)])

    # Customize the plot
    plt.xticks(k_range)
    ax.set_title('Mean Average Recall at K (MAR@K) Comparison')
    ax.set_ylabel('MAR@K')
    ax.set_xlabel('K')
    ax.legend()

    plt.tight_layout()
    plt.show()

def mapk_plot(mapk_scores, model_names, k_range):
    """
    Plots the mean average precision at k for a set of models to compare.
    Parameters
    ----------
    mapk_scores: list of lists
        list of list of map@k scores over k. This list is in same order as model_names
    model_names: list
        list of model names in same order as coverage_scores
    k_range: list
        list or array identifying all k values in order
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert data to dataframe for plotting
    mapk_df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)

    # Plot lines for each model
    for i, model in enumerate(model_names):
        ax.plot(k_range, mapk_df[model], linewidth=3,
                label=model, color=RECOMMENDER_PALETTE[i % len(RECOMMENDER_PALETTE)])

    # Customize the plot
    plt.xticks(k_range)
    ax.set_title('Mean Average Precision at K (MAP@K) Comparison')
    ax.set_ylabel('MAP@K')
    ax.set_xlabel('K')
    ax.legend()

    plt.tight_layout()
    plt.show()

def class_separation_plot(pred_df, n_bins=150, threshold=None, figsize=(10, 6), title=None):
    """
    Plots the predicted class probabilities for multiple classes.
    Parameters
    ----------
    pred_df: pandas dataframe
        a dataframe containing columns named "predicted" and "truth"
    n_bins: number of bins for histogram
    threshold: float, default=None
        A single number between 0 and 1 identifying the threshold to classify observations
    figsize: tuple
        size of figure
    title: str, default=None
        plot title
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique classes
    classes = pred_df.truth.unique()

    # Plot histogram for each class
    for i, single_class in enumerate(classes):
        # Filter data for this class
        class_data = pred_df.query("truth == @single_class")["predicted"]

        # Plot histogram with normal KDE
        ax.hist(class_data, bins=n_bins, alpha=0.5, density=True,
                color=RECOMMENDER_PALETTE[i % len(RECOMMENDER_PALETTE)],
                label=f"True {single_class}")

        # Add KDE manually
        from scipy.stats import gaussian_kde
        if len(class_data) > 1:  # Need at least 2 points for KDE
            x_range = np.linspace(min(class_data), max(class_data), 1000)
            kde = gaussian_kde(class_data)
            ax.plot(x_range, kde(x_range), color=RECOMMENDER_PALETTE[i % len(RECOMMENDER_PALETTE)])

    # Add threshold line if specified
    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle='--')

    # Set labels
    plt.xlabel("Predicted value")
    plt.ylabel("Frequency")
    plt.title(title if title else " ")
    plt.legend()

    plt.tight_layout()
    plt.show()

def roc_plot(actual, model_probs, model_names, figsize=(10, 10)):
    """
    Receiver Operating Characteristic Plot. Can plot multiple models.
    Parameters
    ----------
    actual: array
        array of true classes assignments
    model_probs: list of arrays
        a list containing classification probabilities for each model
    model_names: list or str
        model name(s)
    figsize: tuple
        size of figure
    """
    # Convert to list if not already
    if isinstance(model_names, str):
        model_names = [model_names]
    if not isinstance(model_probs, list):
        model_probs = [model_probs]

    if len(model_names) > 5:
        raise ValueError("Can only compare 5 models or less.")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve for each model
    for m, (probs, name) in enumerate(zip(model_probs, model_names)):
        fpr, tpr, _ = roc_curve(actual, probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, color=RECOMMENDER_PALETTE[m % len(RECOMMENDER_PALETTE)],
                label=f'{name} AUC = {roc_auc:.4f}')

    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'r--')

    # Set labels
    ax.set_title('Receiver Operating Characteristic Plot')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend()

    plt.tight_layout()
    plt.show()

def precision_recall_plot(targs, preds, figsize=(6, 6)):
    """
    Plots the precision recall curve
    Parameters
    ----------
    targs: array-like
        true class labels
    preds: array-like
        predicted probabilities
    figsize: tuple
        size of figure
    """
    # Calculate metrics
    average_precision = average_precision_score(targs, preds)
    precision, recall, _ = precision_recall_curve(targs, preds)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot step curve
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    # Set labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f'2-class Precision-Recall curve: AP={average_precision:.2f}')

    plt.tight_layout()
    plt.show()

def metrics_plot(model_names, coverage_scores, personalization_scores, intra_list_similarity_scores):
    """
    Creates a radar plot showing multiple metrics for multiple models.
    Parameters
    ----------
    model_names: list
        list of model names
    coverage_scores: list
        list of coverage scores in same order as model_names
    personalization_scores: list
        list of personalization scores in same order as model_names
    intra_list_similarity_scores: list
        list of intra-list similarity scores in same order as model_names
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add each model as a trace
        for i, (model_name, coverage, personalization, intra_list_similarity) in enumerate(zip(
            model_names, coverage_scores, personalization_scores, intra_list_similarity_scores
        )):
            fig.add_trace(
                go.Scatterpolar(
                    r=[coverage, personalization * 100, intra_list_similarity * 100],
                    theta=['coverage', 'personalization', 'intra list similarity'],
                    fill='tonext',
                    name=model_name
                )
            )

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True
        )

        fig.show()
    except ImportError:
        print("Plotly is required for metrics_plot. Please install with: pip install plotly")
