D = 32 * (33) // 2  # Dimensionality of the output vector
    phi = np.zeros((challenges.shape[0], D))
    L = challenges.shape[1]

    for i in range(L):
        z_ij = np.cumprod(1 - 2 * challenges[:, i:], axis=1)
        end_index = i * (63 - i) // 2 + (32 - i)
        phi[:, i * (63 - i) // 2: end_index] = z_ij

    return phi