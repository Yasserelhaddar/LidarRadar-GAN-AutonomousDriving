import torch
from scipy.spatial import distance

import numpy as np

def knn(pc,n_neighbors=11):  
    dist = torch.cdist(pc,pc) 
    neigbhors = dist.topk(k=n_neighbors,dim=2,largest=False)
    #print(neigbhors.indices.shape)
    knn_values = pc[:,torch.squeeze(neigbhors.indices[:,:,1:n_neighbors+1])]
    l2 = torch.tensor(0)
    h = 20
    #eps = 1
    for j in range(5):
        l2 = l2 + torch.sum((torch.squeeze(knn_values)[:, j, :] - torch.squeeze(pc))**2)

    #dist2 = torch.max(l2, torch.tensor(eps))
    dist2 = l2
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / h ** 2)

    # uniform_loss = torch.mean((self.radius - dist) * weight)
    uniform_loss = torch.mean((0.01 - dist) * weight) # punet

    return uniform_loss


def distRepuslion(pred):
    
    repulsion_loss = []
    for i in range(pred.shape[0]):
        #print(pred[i,:,:].shape)
        D = distance.squareform(distance.pdist(pred[i,:,:]))
        #print(np.round(D, 1))
        closest = np.argsort(D, axis=1)
        k = 5  # For each point, find the 3 closest points
        h = 5
        eps = 1

        knn_values = pred[i,closest[:, 1:k+1]]

        print(knn_values.shape)
        print(pred[i,:,:].shape)

        l2 = 0
        for j in range(knn_values.shape[1]):
            l2 = l2 + torch.sum(np.power((knn_values[:, j, :] - pred[i, :, :]), 2))
                
        dist2 = torch.max(l2, torch.tensor(eps))
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2**2 / h ** 2)

        # uniform_loss = torch.mean((self.radius - dist) * weight)
        uniform_loss = torch.mean((0.3 - dist) * weight) # punet

        repulsion_loss.append(uniform_loss)
        #repulsion_loss.append(l2)
            
    return torch.Tensor(repulsion_loss)
    


if torch.cuda.is_available():  
    dev = "cuda:0" 

device = torch.device(dev)

print(torch.cuda.get_device_name(0))

points1 = torch.rand(1, 16000, 3)
points2 = torch.rand(1000, 3, requires_grad=True).cuda()

#print("dist: ", distRepuslion(points1).cuda())
print("dist: ", knn(points1).cuda())
