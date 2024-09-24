import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper as itk_wrapper
import itk
import torch
import numpy as np
import torch.nn.functional as F

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda, device="cuda"):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity
        self.device = device

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(self.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(self.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.tensor([[[[delta]], [[0.0]]]]).to(self.device)
            dy = torch.tensor([[[[0.0]], [[delta]]]]).to(self.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(self.device)
            dy = torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(self.device)
            dz = torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(self.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.tensor([[[delta]]]).to(self.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )

    def clean(self):
        del self.phi_AB, self.phi_BA, self.phi_AB_vectorfield, self.phi_BA_vectorfield, self.warped_image_A, self.warped_image_B

def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5), device = "cuda"):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))

    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda, device=device)
    net.assign_identity_map(input_shape)
    return net

def make_sim(similarity):
    if similarity == "LNCC":
        return icon.LNCC(sigma=5)
    elif similarity == "Squared LNCC":
        return icon. SquaredLNCC(sigma=5)
    elif similarity == "MIND-SSC":
        return icon.MINDSSC(radius=2, dilation=2)
    else:
        raise ValueError(f"Similarity measure {similarity} not recognized. Choose from [lncc, lncc2, mind].")

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def apply_mask(image, segmentation):
    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                            itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()
    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                    itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()

def preprocess(image, modality="ct", segmentation=None):
    if modality == "CT/CBCT":
        min_ = -1000
        max_ = 1000
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        image = itk.clamp_image_filter(image, Bounds=(-1000, 1000))
    elif modality == "MRI":
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        min_, _ = itk.image_intensity_min_max(image)
        max_ = quantile(torch.tensor(np.array(image)), .99).item()
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    else:
        raise ValueError(f"{modality} not recognized. Use 'ct' or 'mri'.")

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 

    if segmentation is not None:
        image = apply_mask(image, segmentation)
    return image
    
    
def register_pair(
    model, image_A, image_B, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(model.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(model.device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_resized, B_resized)
    else:
        loss = itk_wrapper.finetune_execute(model, A_resized, B_resized, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        itk_wrapper.create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        itk_wrapper.create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    )
    return itk_transforms

