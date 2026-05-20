class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index
        self.threshold = threshold  # Split value
        self.left = left            # Left child
        self.right = right          # Right child
        self.value = value          # Leaf value


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    # -----------------------------
    # Train the tree
    # -----------------------------
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    # -----------------------------
    # Build tree recursively
    # -----------------------------
    def _grow_tree(self, X, y, depth=0):

        n_samples = len(X)
        n_labels = len(set(y))

        # Stop conditions
        if depth >= self.max_depth or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)

        # Split data
        left_X, left_y, right_X, right_y = self._split(
            X, y, best_feature, best_threshold
        )

        # Recursive tree creation
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    # -----------------------------
    # Find best split
    # -----------------------------
    def _best_split(self, X, y):

        best_gini = float("inf")
        split_idx = None
        split_threshold = None

        n_features = len(X[0])

        for feature in range(n_features):

            thresholds = set(row[feature] for row in X)

            for threshold in thresholds:

                gini = self._gini_split(X, y, feature, threshold)

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    # -----------------------------
    # Calculate Gini Impurity
    # -----------------------------
    def _gini(self, labels):

        counts = {}

        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        impurity = 1

        for count in counts.values():
            prob = count / len(labels)
            impurity -= prob ** 2

        return impurity

    # -----------------------------
    # Gini after split
    # -----------------------------
    def _gini_split(self, X, y, feature, threshold):

        left_y = []
        right_y = []

        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_y.append(y[i])
            else:
                right_y.append(y[i])

        if len(left_y) == 0 or len(right_y) == 0:
            return float("inf")

        left_gini = self._gini(left_y)
        right_gini = self._gini(right_y)

        total = len(y)

        weighted_gini = (
            (len(left_y) / total) * left_gini +
            (len(right_y) / total) * right_gini
        )

        return weighted_gini

    # -----------------------------
    # Split dataset
    # -----------------------------
    def _split(self, X, y, feature, threshold):

        left_X, left_y = [], []
        right_X, right_y = [], []

        for i in range(len(X)):

            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    # -----------------------------
    # Most common class
    # -----------------------------
    def _most_common_label(self, y):

        counts = {}

        for label in y:
            counts[label] = counts.get(label, 0) + 1

        return max(counts, key=counts.get)

    # -----------------------------
    # Predict single sample
    # -----------------------------
    def _traverse_tree(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    # -----------------------------
    # Predict all samples
    # -----------------------------
    def predict(self, X):

        predictions = []

        for x in X:
            predictions.append(self._traverse_tree(x, self.root))

        return predictions