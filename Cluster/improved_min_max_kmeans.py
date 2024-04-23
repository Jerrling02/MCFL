import numpy as np
from copy import deepcopy
from sklearn.exceptions import NotFittedError

class ImprovedMinMaxKMeans:
    def __init__(self, n_clusters=3, p_max=0.5, p_step=0.01, beta=0.1, variance_threshold=10**-6, max_iter=500, verbose=0):
        self.n_clusters = n_clusters
        self.p_max = p_max
        self.p_step = p_step
        self.beta = beta
        self.variance_threshold = variance_threshold
        self.max_iter = max_iter
        self.verbose = verbose
        self.labels_ = None
        self.cost_ = 0.0001
        self.clusters_variance_ = None
        self.cluster_centers_ = None
        self.p_ = 0
        self.weights_ = None
        self.n_iter_ = 0

    def fit(self, X,flag):
        #Validate the parameters
        self.validate_parameters()
        #Initialize cluster centroids
        self.cluster_centers_ = self.initialize_centroids(X)

        #Initialize cluster weights
        self.weights_ = np.asarray([1/self.n_clusters] * self.n_clusters)
        old_weights = np.asarray([1 / self.n_clusters] * self.n_clusters)
        #Initialize cluster assignments
        current_cluster_assignments = np.asarray([[] for _ in range(self.n_clusters)])
        old_cluster_assignments = np.asarray([[] for _ in range(self.n_clusters)])
        t = 0
        p_init = 0
        empty_cluster = False
        self.p_ = p_init
        converged = False
        variance_difference = 0
        times_equal = 0

        while t < self.max_iter and not converged:
            t = t + 1
            current_cluster_assignments = self.update_cluster_assignments(X)
            #Check for empty cluster and update its value
            if self.exists_singleton_cluster(current_cluster_assignments):
                empty_cluster = True
                if self.verbose:
                    print("Empty cluster found")
                self.p_ = self.p_ - self.p_step
                if self.p_ < p_init:
                    if self.verbose:
                        print("p cannot be decreased further")
                        print("Aborting Execution")
                    if flag:
                        self.labels_ = self.get_instances_labels(current_cluster_assignments, X)
                        return
                    else:
                        return None
                #Revert to the assignments and weights corresponding to the reduced p
                current_cluster_assignments = deepcopy(old_cluster_assignments)
                self.weights_ = old_weights.copy()

            #Update cluster centers
            self.update_cluster_centers(current_cluster_assignments, X)

            if self.p_ < self.p_max and not empty_cluster:
                #Store the current assignments in delta(p)
                old_cluster_assignments = deepcopy(current_cluster_assignments)
                #Store the previous weights in vector W(p)
                old_weights = np.copy(self.weights_)
                self.p_ = self.p_ + self.p_step

            #Update the weights
            self.update_weights(current_cluster_assignments, X)
            #Check for convergence
            cost = self.compute_cost()
            converged = np.abs(1 - cost/self.cost_) < self.variance_threshold

            #Extra stop criteria implemented
            if np.abs(variance_difference - np.abs(cost-self.cost_)) < self.variance_threshold:
                times_equal = times_equal + 1
                if times_equal > 3:
                    converged = True
            else:
                times_equal = 0

            if self.verbose:
                print("Iteration ", t, "/", self.max_iter)
                print("Variance difference:", np.abs(cost-self.cost_))
            self.cost_ = cost
        if self.verbose:
            print("Converging for p=", self.p_, "after ", t, " iterations.")
        self.n_iter_ = t
        self.labels_ = self.get_instances_labels(current_cluster_assignments, X)
        return self


    def predict(self, X):
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")
        if self.cluster_centers_ is None:
            raise NotFittedError(msg % {'name': type(self).__name__})

        cluster_assignments = self.update_cluster_assignments(X)
        labels = self.get_instances_labels(cluster_assignments, X)
        return labels

    def fit_predict(self, X,flag):
        self.fit(X,flag)
        return self.labels_


    def compute_cost(self):
        cost = 0
        for k in range(self.n_clusters):
            cost = cost + self.weights_[k]*self.clusters_variance_[k]
        return cost

    def get_instances_labels(self, cluster_assignments, X):
        N = X.shape[0]
        labels = np.zeros(N)
        for cluster, values in enumerate(cluster_assignments):
            labels[values] = cluster
        return labels

    def update_cluster_centers(self, current_cluster_assignments, X):
        #Update cluster centers
        for i in range(self.n_clusters):
            mask = (current_cluster_assignments[i])
            multiplication = np.sum(X[mask], axis=0)
            count = len(mask)
            self.cluster_centers_[i] = multiplication/count if count > 0 else 0

    def update_weights(self, cluster_assignments, X):
        self.clusters_variance_ = self.compute_clusters_variance(cluster_assignments, X)
        total_variance = np.sum(np.power(self.clusters_variance_, 1/(1-self.p_)))
        for k in range(self.n_clusters):
            variance = self.clusters_variance_[k]
            self.weights_[k] = self.beta * self.weights_[k] + (1 - self.beta)*np.power(variance, 1/(1-self.p_))/total_variance


    def compute_clusters_variance(self, cluster_assignments,X):
        #Initialize variance
        variance = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            mask = (cluster_assignments[k])
            variance[k] = np.power(np.linalg.norm(X[mask] - self.cluster_centers_[k]), 2)
        return variance


    def exists_singleton_cluster(self, cluster_assignments):
        #Return true if some cluster is empty or only one instance is found

        for cluster in cluster_assignments:
            if len(cluster) <= 1:
                return True
        return False

    def update_cluster_assignments(self, X):
        # Update the cluster assignments
        new_clusters = [[] for _ in range(self.n_clusters)]
        N = X.shape[0]
        for i in range(N):
            cluster_index = self.compute_minimization_step(X[i])
            new_clusters[cluster_index].append(i)
        return new_clusters


    def compute_minimization_step(self, instance):
        distances = []
        for i in range(self.n_clusters):
            distance = np.power(self.weights_[i], self.p_) * np.power(np.linalg.norm(instance - self.cluster_centers_[i]), 2)
            distances.append(distance)
        return np.argmin(distances)


    def validate_parameters(self):
        if self.max_iter <= 0:
            raise ValueError(
                'Number of iterations should be a positive number,'
                ' got %d instead' % self.max_iter
            )


    def initialize_centroids(self, X):
        centroids_indexs = np.random.choice(range(len(X)), self.n_clusters, replace=False)
        return X[centroids_indexs]



