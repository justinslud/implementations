import numpy as np
import pdb
#pdb.set_trace()

def get_center(points):
    return [points[:, i].mean() for i in range(points.shape[1])]

def get_centroids(X, assignments, k):
    
    centroids = []

    for i in range(k):
        
        points = np.where(assignments == i)[0]
        
        centroids.append(get_center(points))
                         
    return centroids

def get_index(distances):
    index = np.where(distances == np.amax(distances))
            
    if len(index) > 1:
        index = np.random.choice(index)[0]

    else:
        index = index[0][0]
    print(index)
    return index

def kmeans(X, k, iterations=100):

    samples, features = X.shape

    centroids = np.random.random((k, features))

    updated_assignments = np.ones(samples)

    assignments = np.random.choice(range(k), samples)

    iteration = 0

    while sum(updated_assignments == assignments) != samples and iteration < iterations: 
        changes = sum([int(i!=j) for i, j in zip(assignments, updated_assignments)])
        print(iteration, changes)
        assignments = updated_assignments

        updated_assignments = np.ones(samples)

        for i, x in enumerate(X):

            distances = []

            for centroid in centroids:

                distances.append(np.linalg.norm(x - centroid))

            index = get_index(distances)

            updated_assignments[i] = index

        updated_centroids = get_centroids(X, updated_assignments, k)

        iteration += 1

    return centroids


def assign_cluster(Y, centroids):

    assignments = []

    for y in Y:

        distances = []

        for centroid in centroids:

            distances.append(np.linalg.norm(y - centroid))

        index = get_index(distances)

        assignments.append(index)

    return assignments

np.random.seed(803)
X = np.random.random((150, 5))
centroids = kmeans(X, 4)

Y = np.random.random((20, 5))
assignments = assign_cluster(Y, centroids)

print(assignments)

