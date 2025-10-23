def extract_non_diagonal_matrix(input):
    """
    Extract the non-diagonal elements from the input matrix at the last two dimensions.
    input shape: [b, n, n]
    """
    flatten_input = input.reshape([-1, input.shape[-2], input.shape[-1]])
    b, n, _ = flatten_input.shape

    non_diagonal_input = flatten_input.flatten(start_dim=1)[:, 1:]
    non_diagonal_input = non_diagonal_input.view(b, n - 1, n + 1)[:, :, :-1]
    non_diagonal_input = non_diagonal_input.reshape([b, n, n - 1])

    return non_diagonal_input

def split_features(mod_features):
    """
    Split the feature into private space and shared space.
    mod_feature: [b, seq, dim], where we use the sequence sampler
    """
    split_mod_features = {}

    for mod in mod_features:
        if mod_features[mod].ndim == 2:
            split_dim = mod_features[mod].shape[1] // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, 0:split_dim],
                "private": mod_features[mod][:, split_dim:],
            }
        else:
            b, seq, dim = mod_features[mod].shape
            split_dim = dim // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, :, 0:split_dim],
                "private": mod_features[mod][:, :, split_dim : 2 * split_dim],
            }

    return split_mod_features
