#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-19 17:38:07
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import numpy as np
import skimage.measure


def _max_clustering_python(im, im_fgnd_mask, rad):
    sx = im.shape[0]
    sy = im.shape[1]
    r = int(rad + 0.5)

    [py, px] = np.nonzero(im_fgnd_mask)
    num_pixels = py.shape[0]

    local_max_val = np.zeros([sy, sx], dtype=np.float)
    local_max_ind = np.zeros([sy, sx], dtype=np.int)
    peak_found = np.zeros([sy, sx], dtype=np.int32)
    min_im_val = np.min(im)

    for i in range(num_pixels):
        cx = px[i]
        if cx < 0 or cx >= sx:
            continue
        cy = py[i]
        if cy < 0 or cy >= sy:
            continue

        cval = im[cy, cx]
        my = cy
        mx = cx
        mval = cval
        changed = 0
        for ox in range(-r, r + 1):
            nx = cx + ox
            if nx < 0 or nx >= sx:
                continue
            for oy in range(-r, r + 1):
                if (ox * ox + oy * oy) > rad * rad:
                    continue
                ny = cy + oy
                if ny < 0 or ny >= sy:
                    continue
                nval = min_im_val
                if im_fgnd_mask[ny, nx]:
                    nval = im[ny, nx]

                if nval > mval:
                    changed = True
                    mval = nval
                    mx = nx
                    my = ny

        local_max_val[cy, cx] = mval
        local_max_ind[cy, cx] = my * sx + mx
        if not changed:  # this pixel itself is the maximum in its neighborhood
            peak_found[cy, cx] = 1

    # find local peaks of all requested pixels
    maxpath_np = np.zeros([1000, 2], dtype=np.long)
    maxpath = maxpath_np
    path_len = maxpath.shape[0]

    for i in range(num_pixels):
        cx = px[i]
        if cx < 0 or cx >= sx:
            continue
        cy = py[i]
        if cy < 0 or cy >= sy:
            continue

        # initialize tracking trajectory
        end_pos = 0
        end_x = cx
        end_y = cy
        end_ind = cy * sx + cx
        end_max_ind = local_max_ind[end_y, end_x]

        maxpath[end_pos, 0] = end_x
        maxpath[end_pos, 1] = end_y

        while not peak_found[end_y, end_x]:
            # increment trajectory counter
            end_pos += 1

            # if overflow, increase size
            if end_pos >= path_len:
                maxpath_np.resize([path_len * 2, 2])
                maxpath = maxpath_np
                path_len = maxpath.shape[0]

            # add local max to trajectory
            end_ind = end_max_ind
            end_x = int(end_ind % sx)
            end_y = int(end_ind / sx)
            end_max_ind = local_max_ind[end_y, end_x]

            maxpath[end_pos, 0] = end_x
            maxpath[end_pos, 1] = end_y

        for i in range(end_pos + 1):
            cx = maxpath[i, 0]
            cy = maxpath[i, 1]

            local_max_ind[cy, cx] = end_max_ind
            local_max_val[cy, cx] = local_max_val[end_y, end_x]
            peak_found[cy, cx] = 1

    return np.asarray(local_max_val), np.asarray(local_max_ind)


def max_clustering(im_response, im_fgnd_mask, r=10):
    """Local max clustering pixel aggregation for nuclear segmentation.
    Takes as input a constrained log or other filtered nuclear image, a binary
    nuclear mask, and a clustering radius. For each pixel in the nuclear mask,
    the local max is identified. A hierarchy of local maxima is defined, and
    the root nodes used to define the label image.

    Parameters
    ----------
    im_response : array_like
        A filtered-smoothed image where the maxima correspond to nuclear
        center. Typically obtained by constrained-LoG filtering on a
        hematoxylin intensity image obtained from ColorDeconvolution.
    im_fgnd_mask : array_like
        A binary mask of type boolean where nuclei pixels have value
        'True', and non-nuclear pixels have value 'False'.
    r : float
        A scalar defining the clustering radius. Default value = 10.

    Returns
    -------
    im_label : array_like
        im_label image where positive values correspond to foreground pixels that
        share mutual sinks.
    seeds : array_like
        An N x 2 array defining the (x,y) coordinates of nuclei seeds.
    max_response : array_like
        An N x 1 array containing the maximum response value corresponding to
        'seeds'.

    See Also
    --------
    histomicstk.filters.shape.clog

    References
    ----------
    .. [#] XW. Wu et al "The local maximum clustering method and its
       application in microarray gene expression data analysis,"
       EURASIP J. Appl. Signal Processing, volume 2004, no.1, pp.53-63,
       2004.
    .. [#] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
       of Cell Nuclei in Histopathology Images" in IEEE Transactions on
       Biomedical Engineering,vol.57,no.4,pp.847-52, 2010.

    """

    # find local maxima of all foreground pixels
    mval, mind = _max_clustering_python(
        im_response, im_fgnd_mask.astype(np.int32), r
    )

    # identify connected regions of local maxima and define their seeds
    im_label = skimage.measure.label(im_fgnd_mask & (im_response == mval))

    if not np.any(im_label):
        return im_label, None, None

    # compute normalized response
    min_resp = im_response.min()
    max_resp = im_response.max()
    resp_range = max_resp - min_resp

    if resp_range == 0:
        return np.zeros_like(im_label), None, None

    im_response_nmzd = (im_response - min_resp) / resp_range

    # compute object properties
    obj_props = skimage.measure.regionprops(im_label, im_response_nmzd)

    obj_props = [prop for prop in obj_props if np.isfinite(
        prop.weighted_centroid).all()]

    num_labels = len(obj_props)

    if num_labels == 0:
        return im_label, None, None

    # extract object seeds
    seeds = np.array(
        [obj_props[i].weighted_centroid for i in range(num_labels)])
    seeds = np.round(seeds).astype(np.int)

    # fix seeds outside the object region - happens for non-convex objects
    for i in range(num_labels):

        sy = seeds[i, 0]
        sx = seeds[i, 1]

        if im_label[sy, sx] == obj_props[i].label:
            continue

        # find object point with closest manhattan distance to center of mass
        pts = obj_props[i].coords

        ydist = np.abs(pts[:, 0] - sy)
        xdist = np.abs(pts[:, 1] - sx)

        seeds[i, :] = pts[np.argmin(xdist + ydist), :]

        assert im_label[seeds[i, 0], seeds[i, 1]] == obj_props[i].label

    # get seed responses
    max_response = im_response[seeds[:, 0], seeds[:, 1]]

    # set label of each foreground pixel to the label of its nearest peak
    im_label_flat = im_label.ravel()

    pind = np.flatnonzero(im_fgnd_mask)

    mind_flat = mind.ravel()

    im_label_flat[pind] = im_label_flat[mind_flat[pind]]

    # return
    return im_label, seeds, max_response