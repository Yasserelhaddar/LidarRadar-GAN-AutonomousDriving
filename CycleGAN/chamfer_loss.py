import torch


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def batched_pairwise_dist(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    distance = torch.mean(torch.min(P, 2)[0]).float() + torch.mean(torch.min(P, 1)[0]).float()
    total_dist = torch.sum(distance).cuda()
    #return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()
    return total_dist

'''
if torch.cuda.is_available():  
    dev = "cuda:0" 

device = torch.device(dev)

print(torch.cuda.get_device_name(0))

points1 = torch.rand(1, 1000, 3).cuda()
points2 = torch.rand(1, 1000, 3, requires_grad=True).cuda()

print("dist: ", distChamfer(points1, points1).cuda())
'''