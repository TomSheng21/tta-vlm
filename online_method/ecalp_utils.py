import torch
import torch.nn.functional as F

# Define global variables (initialized to None)
sim_im2im, knn_im2im = None, None
sim_im2text, knn_im2text = None, None
sim_im2fewshot, knn_im2fewshot = None, None
sim_im2im_store = None
knn_im2im_store = None



def combine_separate_knns(knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes):
    """Combines KNNs and similarities from image-image and image-text relations."""
    knn_im = knn_im2im + num_classes
    sim_im = sim_im2im

    knn = torch.cat((knn_im, knn_im2text), dim=1)
    sim = torch.cat((sim_im, sim_im2text), dim=1)

    return knn, sim

def combine_separate_knns_fewshot(knn_im2im, sim_im2im, knn_im2text, sim_im2text, knn_fewshot, sim_fewshot,
                                    num_classes, num_fewshot):
    """Combines KNNs and similarities from image-image, image-text and image-fewshot relations."""
    knn_fewshot = knn_fewshot + num_classes
    knn_im = knn_im2im + num_classes + num_fewshot
    sim_im = sim_im2im

    knn = torch.cat((knn_im, knn_im2text, knn_fewshot), dim=1)
    sim = torch.cat((sim_im, sim_im2text, sim_fewshot), dim=1)

    return knn, sim

def sparse_slice(sparse_matrix, row_start, row_end):
    """Slices a sparse matrix along the row dimension."""
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()

    # Filter row indices within the range [row_start, row_end)
    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Adjust row indices to match the new submatrix size
    new_indices[0] = new_indices[0] - row_start

    # Create the new sparse matrix
    new_size = (row_end - row_start, sparse_matrix.size(1))
    new_sparse_matrix = torch.sparse_coo_tensor(new_indices, new_values, new_size)

    return new_sparse_matrix


def edge_reweighting(X, Q, a, k, return_full=False, force_1=False):
    """
    Calculates weighted similarity between two sets of feature vectors.

    Args:
        X (torch.Tensor): Feature vectors of shape (N, D).
        Q (torch.Tensor): Feature vectors of shape (M, D).
        a (torch.Tensor): Another set of feature vectors of shape (A, D) used for weighting.
        k (int): Number of top-k neighbors to return.
        return_full (bool, optional): Whether to return full similarity matrix. Defaults to False.
        force_1 (bool, optional): Force to use weights calculated from `a` only. Defaults to False.

    Returns:
        torch.Tensor: Top-k nearest neighbor indices and similarities if return_full is False,
                       else the full similarity matrix.
    """

    # Calculate feature weights based on variance across dimensions
    

    if (X.shape != Q.shape) and (not force_1):
        # Alternative weighting scheme when shapes differ

        weights = 1 / (X.view(1, -1, a.shape[1]).var(dim=1).mean(dim=0))
        Q_weighted = Q * weights
        X_weighted = X 
    else:
        # Use the first weighting scheme when shapes are the same
        weights = a.var(dim=0)
        weights = F.normalize(weights, p=2, dim=0)
        Q_weighted = Q * weights
        X_weighted = X

    # Normalize weighted feature vectors
    Q_weighted = F.normalize(Q_weighted, p=2, dim=1)
    X_weighted = F.normalize(X_weighted, p=2, dim=1)

    # Calculate weighted similarity matrix
    weighted_similarity = Q_weighted @ X_weighted.t()

    if return_full:
        return weighted_similarity

    # Find top-k nearest neighbors
    s, knn = weighted_similarity.topk(k, largest=True, dim=1)

    return knn, s


def knn_to_weighted_adj_matrix(knn, s, num_nodes):
    """
    Converts k-nearest neighbor indices and similarities to a sparse weighted adjacency matrix.

    Args:
        knn (torch.Tensor): k-nearest neighbor indices of shape (num_nodes, k).
        s (torch.Tensor): Similarities corresponding to the k-nearest neighbors of shape (num_nodes, k).
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.sparse_coo_tensor: Sparse weighted adjacency matrix.
    """

    # Flatten the knn and s arrays
    knn_indices = knn.flatten()
    distances = s.flatten()

    # Replace -1 indices (padding) with 0
    knn_indices[knn_indices == -1] = 0

    # Create row indices
    row_indices = torch.arange(num_nodes, device=knn.device).repeat_interleave(knn.shape[1])

    # Filter out zero distances
    nonzero_mask = distances != 0
    filtered_row_indices = row_indices[nonzero_mask]
    filtered_knn_indices = knn_indices[nonzero_mask]
    filtered_distances = distances[nonzero_mask]

    # Create the sparse weighted adjacency matrix
    W = torch.sparse_coo_tensor(
        torch.stack([filtered_row_indices, filtered_knn_indices]),
        filtered_distances,
        (num_nodes, num_nodes)
    )

    return W


def dynamic_graph_expansion(features, clf, fewshot_features, k_text, k_image, k_fewshot):
    """
    Creates graph edges with separate KNN computations for image-image, image-text, and image-fewshot relations.
    """
    global sim_im2im, knn_im2im, sim_im2text, knn_im2text, sim_im2fewshot, knn_im2fewshot, sim_im2im_store, knn_im2im_store

    num_classes = clf.shape[0]  # Number of text features
    num_features = features.shape[0]  # Number of image features

    # Determine if this is the first time creating the graph
    first_time = (sim_im2im is None) or (features.shape[0] == 2)

    # Determine number of fewshot examples
    num_fewshot = fewshot_features.shape[0] if fewshot_features is not None else 0

    # Limit KNN based on number of available features
    k_im2im = min(k_image, features.shape[0])


    # Calculate image-image similarities
    if first_time:
        # Calculate full similarity matrix
        sim_full = edge_reweighting(
            features, features, clf, k=k_im2im, return_full=True, force_1=True
        ).float()
        sim_im2im, knn_im2im = sim_full[:num_features, :num_features].topk(k_im2im, largest=True, dim=1)
        # Store for subsequent calls
        sim_im2im_store = sim_im2im.clone()
        knn_im2im_store = knn_im2im.clone()

    else: # It is not the first time
        # Calculate similarity only for the last two image features (incremental update)
        sim_full_cr = edge_reweighting(
            features[-2:], features, clf, k=k_im2im, return_full=True, force_1=True
        ).t().float()

        # Update KNN and similarity incrementally by using the stored values and the newly calculated ones
        sim_new, knn_new = sim_full_cr.topk(k_im2im, largest=True, dim=1)      # Top k similarities and indices for the new nodes
        knn_new_idx = torch.full((knn_im2im.shape[0] - 1, 1), num_features - 1,  # Indices for the KNN of the new nodes
                                  dtype=knn_new.dtype).to(knn_new.device)

        # Ensure KNNs are not compared with the same node and invalidate old entries
        last_mask = knn_im2im_store == num_features - 2
        sim_im2im_store[last_mask] = -1.0

        # Concatenate existing and new KNNs, extend similarity and KNN info
        sim_extended = torch.cat([sim_im2im_store[:-1], sim_full_cr[:, :num_features - 2].transpose(1, 0)],
                                 dim=1)
        knn_extended = torch.cat([knn_im2im_store[:-1], knn_new_idx - 1, knn_new_idx], dim=1)

        # Preserve only the best (highest similarity score)
        if sim_extended.shape[0] == k_im2im + 1:
            k_im_new = k_im2im + 1
        elif sim_extended.shape[0] > k_im2im:
            k_im_new = k_im2im + 2
        else:
            k_im_new = k_im2im

        # Truncate redundant
        sim_im2im_store, ht = sim_extended.topk(k_im_new, largest=True, dim=1)
        sim_im2im = sim_im2im_store[:, :k_im2im]
        knn_im2im_store = knn_extended.gather(1, ht)
        knn_im2im = knn_im2im_store[:, :k_im2im]

        if sim_extended.shape[0] > k_im2im:
            sim_new_more, knn_new_more = sim_full_cr.topk(k_im_new, largest=True, dim=1)
            sim_im2im_store = torch.cat([sim_im2im_store, sim_new_more], dim=0)
            knn_im2im_store = torch.cat([knn_im2im_store, knn_new_more], dim=0)
        else:
            sim_im2im_store = torch.cat([sim_im2im_store, sim_new], dim=0)
            knn_im2im_store = torch.cat([knn_im2im_store, knn_new], dim=0)

        # Concatenate the newest
        sim_im2im = torch.cat([sim_im2im, sim_new], dim=0)
        knn_im2im = torch.cat([knn_im2im, knn_new], dim=0)

    # Limit KNN based on number of classes
    k_im2text = min(k_text, num_classes)

    # Calculate image-text similarities
    if first_time:
        sim_full_text = features.mm(clf.t())
        
        sim_im2text, knn_im2text = sim_full_text[:num_features].topk(k_im2text, largest=True, dim=1)

    else:
        sim_full_text_cr = features[-1].unsqueeze(0).mm(clf.t())  # Only calculate for newest
        sim_new, knn_new = sim_full_text_cr.topk(k_im2text, largest=True, dim=1) # Get top k
        sim_im2text = torch.cat((sim_im2text[:-1], sim_new, sim_new), axis=0)   # Concatenate to known
        knn_im2text = torch.cat((knn_im2text[:-1], knn_new, knn_new), axis=0)   # Ditto

    # Handle few-shot image features
    if fewshot_features is not None:
        k_fewshot = min(k_fewshot, fewshot_features.shape[0])
        if first_time:
            sim_full_fewshot = edge_reweighting(
                fewshot_features, features, clf, k=k_fewshot, return_full=True, force_1=False
            )
        else:
            sim_full_fewshot_cr = edge_reweighting(
                fewshot_features, features[-1:], clf, k=k_fewshot, return_full=True, force_1=False
            )

        if first_time:
            sim_im2fewshot, knn_im2fewshot = sim_full_fewshot[:num_features].topk(k_fewshot, largest=True, dim=1)
        else:
            sim_new, knn_new = sim_full_fewshot_cr.topk(k_fewshot, largest=True, dim=1) # Similar KNNs
            sim_im2fewshot = torch.cat((sim_im2fewshot[:-1], sim_new, sim_new), axis=0)     # Append newest
            knn_im2fewshot = torch.cat((knn_im2fewshot[:-1], knn_new, knn_new), axis=0)     # More KNNs

        # Combine all KNN sources: image, text, fewshot
        knn, sim = combine_separate_knns_fewshot(
            knn_im2im, sim_im2im, knn_im2text, sim_im2text, knn_im2fewshot, sim_im2fewshot, num_classes, num_fewshot
        )

        # Apply offsets to account for different feature spaces
        # Initialize similarity for text and few-shot samples
        knn_fewshot = -1 * torch.ones((num_fewshot, knn.shape[1]), dtype=knn.dtype).to(knn.device)
        sim_fewshot = torch.zeros((num_fewshot, sim.shape[1]), dtype=sim.dtype).to(sim.device)

        knn_text = -1 * torch.ones((num_classes, knn.shape[1]), dtype=knn.dtype).to(knn.device)
        sim_text = torch.zeros((num_classes, sim.shape[1]), dtype=sim.dtype).to(sim.device)

        # Concatenate all
        knn = torch.cat((knn_text, knn_fewshot, knn), axis=0)
        sim = torch.cat((sim_text, sim_fewshot, sim), axis=0)

    else:  # No few-shot features
        knn, sim = combine_separate_knns(
            knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes
        )

        # Append zeroed sim and -1 knn info, and offset KNN info to ensure uniqueness
        knn_text = -1 * torch.ones((num_classes, knn.shape[1]), dtype=knn.dtype).to(knn.device)
        sim_text = torch.zeros((num_classes, sim.shape[1]), dtype=sim.dtype).to(sim.device)
        knn = torch.cat((knn_text, knn), axis=0)
        sim = torch.cat((sim_text, sim), axis=0)

    return knn, sim

def iterative_label_propagation(image_features, text_features, k_text, k_image,
                                                    k_fewshot, gamma, alpha, max_iter=3, fewshot_image_features=None,
                                                    fewshot_labels=None):
    """
    Performs iterative transductive label propagation.

    Args:
        image_features (torch.Tensor): Image feature vectors of shape (N, D).
        text_features (torch.Tensor): Text feature vectors (class embeddings) of shape (C, D).
        k_text (int): Number of nearest text neighbors to consider.
        k_image (int): Number of nearest image neighbors to consider.
        k_fewshot (int): Number of nearest fewshot neighbors to consider.
        gamma (float): Scaling factor for image-image similarity.
        alpha (float): Weight for combining propagated labels.
        max_iter (int, optional): Number of label propagation iterations. Defaults to 3.
        fewshot_image_features (torch.Tensor, optional): Few-shot image feature vectors. Defaults to None.
        fewshot_labels (torch.Tensor, optional): Few-shot labels. Defaults to None.

    Returns:
        torch.Tensor: Predicted labels for the image features, along with timing information.
    """
    global Y_0, Y_txt # Initialized label matrices

    if image_features.shape[0] == 2:
        Y_0 = None
        Y_txt = None


    if image_features.dim() == 3:
        image_features = image_features.mean(dim=1)
    num_class = text_features.shape[0]

    if fewshot_image_features is not None:

        num_fewshot = fewshot_image_features.shape[0]

        # Construct base KNN
        knn, sim = dynamic_graph_expansion(image_features, text_features, fewshot_image_features, k_text, k_image, k_fewshot)

        # Create masks for different types of nearest neighbors, different scaling factors apply
        mask_knn = (knn < num_class) & (knn >= 0) # KNNs that falls within the classes
        mask_knn_fewshot = (knn < num_class + num_fewshot) & (knn >= num_class) # KNNs that are in the fewshot range
        sim[mask_knn_fewshot] = sim[mask_knn_fewshot] ** gamma  # Scaling
        sim[mask_knn] = sim[mask_knn] ** gamma                              # Scaling

        # Scale last type
        mask_im2im = (knn >= num_class + num_fewshot)
        sim[mask_im2im] = sim[mask_im2im] ** gamma

    else: # If there are no fewshot img features, treat the problem as single data modality KNNs

        knn, sim = dynamic_graph_expansion(image_features, text_features, None, k_text, k_image, 0)
        num_fewshot = 0                   # Set to 0, in case that the code crashed before and this var is not
        mask_knn = knn < num_class        # KNNs that falls within the classes
        sim[mask_knn] = sim[mask_knn] ** gamma # Apply scaling to classes  
        mask_im2im = (knn>= num_class)    # Indexes above classes, i.e. indexes to images
        sim[mask_im2im] = sim[mask_im2im] ** gamma    # Apply scaling

    W = knn_to_weighted_adj_matrix(knn, sim, image_features.shape[0]+num_class+num_fewshot) # Get weighted adjacency matrix

    W = W.transpose(0,1) +W # Symmetrize the weight matrix

    W_sum = torch.sparse.sum(W, dim=1).to_dense() # Sum of weights
    diag = 1 / (torch.sqrt(W_sum) + 1e-6)          # Invert the weight matrix's diag elements and offset its value by a small amount
    diag = diag.to(W.device)                       # Enforce device consistency

    indices = torch.arange(W.size(0), device=W.device)                   # Indices of diagonal elements
    D_rootsquare = torch.sparse_coo_tensor(                               # Build an inverted square-rooted diagonal weight matrix
        torch.stack([indices, indices]), diag, (W.size(0), W.size(0))     # Pass diag indices and values
    )

    W_tilde = torch.sparse.mm(D_rootsquare, torch.sparse.mm(W, D_rootsquare))   # Final symmetric normalized weight matrix

    if Y_txt is None or Y_txt.shape[0] != num_class+num_fewshot: # Initial label generation
        Y_txt = torch.eye(                  # Identity matrix, 1 means the node is of that category
            num_class,                      # Num classes
            dtype=torch.float32,            # Precision
            device=W.device                # Same device
        ).to_sparse()                       # Conserve memory, matrix is mostly zeroes due to 1 class per sample

        if fewshot_image_features is not None: #"Hot encode" fewshot labels
            Y_fewshot = (fewshot_labels).to_sparse()    # 1 at column indicated by class and 0 elsewhere for each "hot encoded" sample
            Y_txt = torch.cat((Y_txt, Y_fewshot), dim=0)    # Join to previously known classes
            Y_0 = torch.zeros(                         # Initial value assignment for samples
                (W_tilde.size(0) - num_class - num_fewshot, num_class), # Assign for however many samples are present and ensure class is coherent
                dtype=torch.float32,                     # Same precision
                device=W.device                        # Device
            ).to_sparse()                                  # Conserve memory

            Y_hat = torch.cat((Y_txt, Y_0), dim=0)     # Place both types in same matrix

        else:
            Y_0 = torch.zeros((W_tilde.size(0) - num_class - num_fewshot, num_class), dtype=torch.float32, device=W.device).to_sparse()
            Y_hat = torch.cat((Y_txt, Y_0), dim=0)
    else:
        Y_0 = torch.cat((Y_0[:-1], torch.zeros((2, num_class), dtype=torch.float32, device=W.device)), dim=0).to_sparse()
        Y_hat = torch.cat((Y_txt, Y_0), dim=0)

    W_tilde_0 = sparse_slice(W_tilde.coalesce(), num_class+num_fewshot, W_tilde.size(0)) # Filter for label propagation

    # Perform iterative label propagation
    for m in range(max_iter):

        Y_t = alpha * torch.sparse.mm(W_tilde_0 , Y_hat) + (1-alpha)*Y_0 # Propagate scores from neighbors

        Y_hat = torch.cat((Y_txt, Y_t), dim=0) # The new estimation matrix is the Y estimate concatted to the known labels

    pred = Y_hat[-1,:num_class].to_dense()                # Copy estimations

    # Get dim for indexing properly
    tensor = Y_hat.to_dense()[num_class+num_fewshot:,:num_class]  # Retrieve tensor
    top3_values, top3_indices = torch.topk(tensor, 1, dim=1)      # The k highest values of the Y_hat slice: i.e. the columns indicating the class estimations
    Y_0 = torch.zeros_like(tensor)   # Replace previous estimations with what we want to pass on this iteration: i.e. a small amount of the confidence

    Y_0.scatter_(1, top3_indices, top3_values/5)    #  Copy most confident output on to estimations


    return torch.Tensor(pred*10000).unsqueeze(0).to(image_features.device).softmax(dim=1)