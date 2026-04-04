import torch


def total_variation_loss(img):
    """
    Total Variation Loss - penalizes large differences between neighboring pixels.
    Reduces color artifacts and smooths transitions in uniform areas (sky, snow, walls).
    """
    batch_size = img.size()[0]
    h_x = img.size()[2]
    w_x = img.size()[3]
    count_h = (img.size()[2] - 1) * img.size()[3]
    count_w = img.size()[2] * (img.size()[3] - 1)
    h_tv = torch.pow((img[:,:,1:,:] - img[:,:,:h_x-1,:]), 2).sum()
    w_tv = torch.pow((img[:,:,:,1:] - img[:,:,:,:w_x-1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def contrast_loss(fake_img, real_img):
    """
    Contrast Loss - encourages the generator to match the contrast level of the ground truth.
    Useful for scenes with extreme lighting (night scenes, bright outdoor, etc.).
    Measures contrast as the standard deviation of the L channel.
    """
    fake_L = fake_img[:, 0:1, :, :]  # L channel
    real_L = real_img[:, 0:1, :, :]  # L channel

    fake_std = torch.std(fake_L.view(fake_L.size(0), -1), dim=1)  # contrast measure
    real_std = torch.std(real_L.view(real_L.size(0), -1), dim=1)

    return torch.abs(fake_std - real_std).mean()
