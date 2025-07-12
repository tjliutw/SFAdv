import torch

# Define the squared distance covariance (dCov^2) loss/function between two groups of vectors.
class dCov2(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(dCov2, self).__init__()
        self.device = device
    # end of __init__()

    def forward(self, x, y):
        if len(x.size()) == 3:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1)
        else:
            pass

        # distance matrices for x and y, respectively
        x1 = x.unsqueeze(0)
        x2 = x.unsqueeze(1)
        dist_mat_x = (x1 - x2).norm(2, dim=2)

        y1 = y.unsqueeze(0)
        y2 = y.unsqueeze(1)
        dist_mat_y = (y1 - y2).norm(2, dim=2)

        cent_dist_x = dist_mat_x - torch.mean(dist_mat_x, 1).view(-1, 1) - torch.mean(dist_mat_x, 0).view(1, -1) + dist_mat_x.view(-1).mean(0)
        cent_dist_y = dist_mat_y - torch.mean(dist_mat_y, 1).view(-1, 1) - torch.mean(dist_mat_y, 0).view(1, -1) + dist_mat_y.view(-1).mean(0)

        dCov2_value = (cent_dist_x * cent_dist_y).view(-1).mean(0)

        return dCov2_value
    # end of forward()
# end of class dCov2

# Define the cross covariance (XCov) loss/function between two groups of vectors.
class XCov(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(XCov, self).__init__()
        self.device = device
    # end of __init__()

    def forward(self, x, y):
        batch_size = x.size()[0]
        if len(x.size()) == 3:
            x = x.view(batch_size, -1)
        else:
            pass

        diff_x = x - x.mean(0)
        diff_y = y - y.mean(0)

        XCov_value = torch.sum(torch.mm(torch.t(diff_x), diff_y)**2) / (2 * batch_size**2)

        return XCov_value
    # end of forward()
# end of class XCov