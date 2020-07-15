# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
from wbia.constants import ANNOTATION_TABLE
from wbia import dtool as dt
import utool as ut
import numpy as np
import os
import json


(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']

_PLUGIN_FOLDER = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG = os.path.join(_PLUGIN_FOLDER, 'configs/manta.json')
_DEFAULT_CONFIG_DICT = {'config_path': _DEFAULT_CONFIG}


@register_ibs_method
def pie_embedding_timed(ibs, aid_list, config_path=_DEFAULT_CONFIG, use_depc=True):
    import time

    start = time.time()
    ans = ibs.pie_embedding(aid_list, config_path, use_depc)
    elapsed = time.time() - start
    print('Computed %s embeddings in %s seconds' % (len(aid_list), elapsed))
    per_embedding = elapsed / len(aid_list)
    print('average time is %s per embedding' % per_embedding)
    return ans


# note: an embedding is 256xfloat8, aka 2kb in size (using default config)
@register_ibs_method
def pie_embedding(ibs, aid_list, config_path=_DEFAULT_CONFIG, use_depc=True):
    r"""
    Generate embeddings using the Pose-Invariant Embedding (PIE) algorithm made by Olga
    Moskvyak and released on https://github.com/olgamoskvyak/reid-manta.
    Olga's code has been modified and plugin-ified for this work.

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids specifying the input
        config_path (str): path to a PIE config .json file that parameterizes the model
            and directs PIE to the weight file, among other fields

    CommandLine:
        python -m wbia_pie._plugin pie_embedding

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia_pie
        >>> import numpy as np
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> aids = ibs.get_valid_aids(species='Mobula birostris')
        >>> embs_depc    = np.array(ibs.pie_embedding(aids, use_depc=True))
        >>> embs_no_depc = np.array(ibs.pie_embedding(aids, use_depc=False))
        >>> diffs = np.abs(embs_depc - embs_no_depc)
        >>> assert diffs.max() < 1e-8
        >>> # each embedding is 256 floats long so we'll just check a bit
        >>> result = embs_depc[0][:20]
        array([-0.03839333,  0.01182338,  0.02393869, -0.07164327, -0.04367629,
               -0.00150531,  0.01324393,  0.10909598,  0.02349781,  0.08439559,
               -0.06415793,  0.0110384 ,  0.03897202, -0.11256221,  0.00709192,
                0.10403764,  0.00615681, -0.10405623,  0.0320793 , -0.0394897 ])

    Example:
        >>> # ENABLE_DOCTEST
        >>> # This tests that an aid's embedding is independent of the other aids processed in the same call
        >>> import wbia_pie
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> aids = ibs.get_valid_aids(species='Mobula birostris')
        >>> aids1 = aids[:-1]
        >>> aids2 = aids[1:]
        >>> embs1 = ibs.pie_compute_embedding(aids1)
        >>> embs2 = ibs.pie_compute_embedding(aids2)
        >>> # just look at the overlapping aids/embs
        >>> embs1 = np.array(embs1[1:])
        >>> embs2 = np.array(embs2[:-1])
        >>> compare_embs = np.abs(embs1 - embs2)
        >>> compare_embs.max() < 1e-8
        True

    """
    if use_depc:
        config = {'config_path': config_path}
        embeddings = ibs.depc_annot.get(
            'PieEmbedding', aid_list, 'embedding', config=config
        )
    else:
        embeddings = pie_compute_embedding(ibs, aid_list, config_path=config_path)
    return embeddings


class PieEmbeddingConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('config_path', _DEFAULT_CONFIG),
    ]


@register_preproc_annot(
    tablename='PieEmbedding',
    parents=[ANNOTATION_TABLE],
    colnames=['embedding'],
    coltypes=[np.ndarray],
    configclass=PieEmbeddingConfig,
    fname='pie',
    chunksize=128,
)
@register_ibs_method
def pie_embedding_depc(depc, aid_list, config=_DEFAULT_CONFIG_DICT):
    ibs = depc.controller
    embs = pie_compute_embedding(ibs, aid_list, config_path=config['config_path'])
    for aid, emb in zip(aid_list, embs):
        yield (np.array(emb),)


# TODO: delete the generated files in dbpath when we're done computing embeddings
@register_ibs_method
def pie_compute_embedding(
    ibs, aid_list, config_path=_DEFAULT_CONFIG, output_dir=None, prefix=None, export=False
):
    preproc_dir = ibs.pie_preprocess(aid_list)
    from .compute_db import compute

    embeddings, filepaths = compute(preproc_dir, config_path, output_dir, prefix, export)
    embeddings = fix_pie_embedding_order(ibs, embeddings, aid_list, filepaths)
    return embeddings


def fix_pie_embedding_order(ibs, embeddings, aid_list, filepaths):
    filepaths = [_get_parent_dir_and_fname_only(fpath) for fpath in filepaths]
    # PIE messes with extensions, so throw those away
    filepaths = [os.path.splitext(fp)[0] for fp in filepaths]

    names = ibs.get_annot_name_texts(aid_list)
    fnames = ibs.get_annot_image_paths(aid_list)
    fnames = [os.path.split(fname)[1] for fname in fnames]
    aid_filepaths = [os.path.join(name, fname) for name, fname in zip(names, fnames)]
    # PIE messes with extensions, so throw those away
    aid_filepaths = [os.path.splitext(fp)[0] for fp in aid_filepaths]

    # aid_filepaths and filepaths have the same entries in different orders
    filepath_to_idx = {filepaths[i]: i for i in range(len(filepaths))}

    def sorted_embedding(i):
        key = aid_filepaths[i]
        idx = filepath_to_idx[key]
        emb = embeddings[idx]
        return emb

    sorted_embs = [sorted_embedding(i) for i in range(len(aid_list))]
    return sorted_embs


def _get_parent_dir_and_fname_only(fpath):
    parent_path, fname = os.path.split(fpath)
    parent_dir = os.path.split(parent_path)[1]
    parent_dir_and_fname = os.path.join(parent_dir, fname)
    return parent_dir_and_fname


# This function calls the PIE image preprocessing method preproc_db.preproc,
# whose input is a folder full of pre-cropped images. preproc then resizes all
# images to dimensions specified in the config file and organizes the resized
# images in sub-folders for each label (name). These folders will be read by
# PIE's embedding-compute function later.
@register_ibs_method
def pie_preprocess(ibs, aid_list, config_path=_DEFAULT_CONFIG):
    output_dir = pie_preproc_dir(aid_list, config_path)
    label_file_path = os.path.join(output_dir, 'name_map.csv')
    label_file = ibs.pie_name_csv(aid_list, fpath=label_file_path)
    img_path = ibs.imgdir
    from .preproc_db import preproc

    dbpath = preproc(img_path, config_path, lfile=label_file, output=output_dir)
    return dbpath


# pie's preprocess works on every image in a folder, so we put 'em in a folder
def pie_preproc_dir(aid_list, config_path=_DEFAULT_CONFIG):
    # hash the aid_list to make a unique folder name
    unique_prefix = str(hash(tuple(aid_list)))
    with open(config_path) as config_file:
        config = json.loads(config_file.read())
    conf_output_dir = config['prod']['output']
    output_dir = os.path.join(_PLUGIN_FOLDER, conf_output_dir, unique_prefix)
    if not os.path.isdir(output_dir):
        print('PIE preproc_dir creating output directory %s' % output_dir)
        os.makedirs(output_dir)
    print('PIE preproc_dir for aids %s returning %s' % (aid_list, output_dir))
    return output_dir


# PIE's preproc and embed funcs require a .csv file linking filnames to labels (names)
@register_ibs_method
def pie_name_csv(ibs, aid_list, fpath=None):
    if fpath is None:
        fpath = os.path.join(_PLUGIN_FOLDER, 'examples/dev/name_map.csv')
    name_texts = ibs.get_annot_name_texts(aid_list)
    fnames = ibs.get_annot_image_paths(aid_list)
    # only want final, file part of fpaths
    fnames = [fname.split('/')[-1] for fname in fnames]
    csv_dicts = [{'file': f, 'label': l} for (f, l) in zip(fnames, name_texts)]
    _write_csv_dicts(csv_dicts, fpath)
    print('Saved PIE name file to %s' % fpath)
    return fpath


def _write_csv_dicts(csv_dicts, fpath):
    import csv

    keys = csv_dicts[0].keys()
    with open(fpath, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_dicts)


class PieConfig(dt.Config):  # NOQA
    def get_param_info_list(self):
        return [
            ut.ParamInfo('config_path', _DEFAULT_CONFIG),
        ]


class PieRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = False
    _tablename = 'Pie'


@register_preproc_annot(
    tablename='Pie',
    parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'],
    coltypes=[float],
    configclass=PieConfig,
    requestclass=PieRequest,
    fname='pie',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_pie(depc, qaid_list, daid_list, config):
    ibs = depc.controller

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    assert len(qaids) == 1
    qaid = qaids[0]

    # TODO: double-check config_path arg below is right vis à vis depc stuff
    name_dist_dicts = ibs.pie_predict_light(
        qaid, daids, config_path=config['config_path']
    )

    # TODO: below funcs
    name_score_dicts = distance_dicts_to_score_dicts(name_dist_dicts)
    aid_score_list = aid_scores_from_name_scores(ibs, name_score_dicts, daids)

    for qaid, daid, daid_score in zip(qaid_list, daid_list, aid_score_list):
        yield daid_score


def distance_to_score(distance):
    score = 1 / (1 + distance)
    return score


def distance_dicts_to_score_dicts(distance_dicts, conversion_func=distance_to_score):
    score_dicts = distance_dicts.copy()
    for entry in score_dicts:
        entry['score'] = conversion_func(entry['distance'])
    return score_dicts


# We get a score per-name, but now we need to compute scores per-annotation. Done simply by averaging the name score over all of that name's annotations
@register_ibs_method
def aid_scores_from_name_scores(ibs, name_score_dicts, daid_list):
    daid_name_list = ibs.get_annot_name_texts(daid_list)
    # name_score_dict is a list of dicts; we want one dict with names ('label') as keys
    name_info_dict = {dct['label']: dct for dct in name_score_dicts}
    # calculate annotwise score by dividing namescore by # of annots with that name
    for name in name_info_dict.keys():
        count = daid_name_list.count(name)
        name_info_dict[name]['count'] = count
        name_info_dict[name]['annotwise_score'] = name_info_dict[name]['score'] / count

    # the rest of this method could be one big list comprehension; this is for readability
    def compute_annot_score(i):
        name = daid_name_list[i]
        if name in name_info_dict:
            score = name_info_dict[name]['annotwise_score']
        else:
            score = 0.0
        return score

    daid_scores = [compute_annot_score(i) for i in range(len(daid_list))]
    return daid_scores


@register_ibs_method
def pie_predict_light(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG):
    r"""
    Matches an annotation using PIE, by calling PIE's k-means distance measure on PIE embeddings.

    Args:
        ibs (IBEISController): IBEIS / WBIA controller object
        qaid            (int): query annot
        daid_list       (int): database annots
            and directs PIE to the weight file, among other fields
        config_path (str): path to a PIE config .json file that parameterizes the model
            and directs PIE to the weight file, among other fields

    CommandLine:
        python -m wbia_pie._plugin pie_predict_light:0

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia_pie
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> qaid = 2  # name = candy
        >>> daids = [1,3,4,5]
        >>> result = ibs.pie_predict_light(qaid, daids)
        [{'distance': 0.9821137124475428, 'label': 'candy'},
         {'distance': 1.1391638476158368, 'label': 'valentine'},
         {'distance': 1.3051054116154164, 'label': 'jel'},
         {'distance': 1.378156086911126, 'label': 'april'}]

    Example:
        >>> # ENABLE_DOCTEST
        >>> # Test that pie_predict_light and pie_predict return the same results
        >>> import wbia_pie
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> qaid = 2
        >>> daids = [1,3,4,5]
        >>> pred_light = ibs.pie_predict_light(qaid, daids)
        >>> pred       = ibs.pie_predict(qaid, daids)
        [{'distance': 0.7318770335113057, 'label': 'april'},
        {'distance': 1.257755410232425, 'label': 'jel'},
        {'distance': 1.378156086911126, 'label': 'valentine'},
        {'distance': 1.575796497115955, 'label': 'candy'}]

    """
    # just call embeddings once bc of significant startup time on PIE's bulk embedding-generator
    all_aids = daid_list + [qaid]
    all_embs = ibs.pie_embedding(all_aids)

    # now get the embeddings into the shape and type PIE expects
    db_embs = np.array(all_embs[:-1])
    query_emb = np.array(all_embs[-1:])  # query_emb.shape = (1, 256)
    db_labels = np.array(ibs.get_annot_name_texts(daid_list))

    from .predict import pred_light

    ans = pred_light(query_emb, db_embs, db_labels, config_path)
    return ans


@register_ibs_method
def pie_predict_light_2(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG):
    db_embs = np.array(ibs.pie_embedding(daid_list))
    db_labels = np.array(ibs.get_annot_name_texts(daid_list))
    # todo: cache this
    query_emb = ibs.pie_compute_embedding([qaid])
    from .predict import pred_light

    ans = pred_light(query_emb, db_embs, db_labels, config_path)
    return ans


# PIE's predict.predict function is more of a command-line program than a package function. It works off of .csv files, .json files, and images stored in hierachical folders, and does all sorts of preprocessing assuming the qaid embedding hasn't been computed before (so ofc, it computes that). This method is basically deprecated and was a proof-of-concept/MVP, and pie_predict_light should be used in practice. Kept around so we can verify that pie_predict_light produces the same results as the released predict function.
@register_ibs_method
def pie_predict(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG, display=False):
    config = ibs.pie_predict_prepare_config(daid_list, _DEFAULT_CONFIG)
    impath = ibs.get_annot_image_paths(qaid)

    from .predict import predict

    ans = predict(impath, config, config_path, display)
    return ans


# This func modifies a base PIE config file, which contains network parameters as well as database and image paths, keeping the network parameters but updating db/image paths to correspond to daid_list
@register_ibs_method
def pie_predict_prepare_config(ibs, daid_list, base_config_file=_DEFAULT_CONFIG):

    pred_data_dir = ibs.pie_ensure_predict_datafiles(daid_list)

    with open(base_config_file) as conf_file:
        config = json.loads(conf_file.read())

    config['prod']['embeddings'] = pred_data_dir
    config['prod']['temp'] = pred_data_dir

    return config


# pie_predict requires embeddings stored in a .csv file that is identified in the config file, as well as a file linking image names to labels (names). This func makes and saves those csvs if necessary, in a unique folder for a given daid_list.
@register_ibs_method
def pie_ensure_predict_datafiles(ibs, daid_list, base_config_file=_DEFAULT_CONFIG):

    pred_data_dir = pie_annot_info_dir(daid_list)
    embs_fname = os.path.join(pred_data_dir, '_emb.csv')
    lbls_fname = os.path.join(pred_data_dir, '_lbl.csv')

    if not os.path.isfile(embs_fname):
        embeddings = ibs.pie_embedding(daid_list)
        embs_fname = _write_embeddings_csv(embeddings, embs_fname)
    if not os.path.isfile(lbls_fname):
        lbls_fname = _write_labels_csv(ibs, daid_list, lbls_fname)

    return pred_data_dir


# directory where we'll store embeddings and label .csv's to be read by PIE
def pie_annot_info_dir(aid_list):
    embeddings_dir = os.path.join(_PLUGIN_FOLDER, 'embeddings')
    unique_label = str(hash(tuple(aid_list)))
    output_dir = os.path.join(embeddings_dir, unique_label)
    if not os.path.isdir(output_dir):
        print('PIE embeddings_dir creating output directory %s' % output_dir)
        os.makedirs(output_dir)
    return output_dir


def _write_embeddings_csv(embeddings, fname):
    ncols = len(embeddings[0])
    header = ['emb_' + str(i) for i in range(ncols)]
    header = ','.join(header)
    np.savetxt(fname, embeddings, delimiter=',', newline='\n', header=header)
    print('PIE wrote embeddings csv to %s' % fname)
    return fname


def _write_labels_csv(ibs, aid_list, fname):
    names = ibs.get_annot_name_texts(aid_list)
    # PIE expects a zero-indexed "class" column that corresponds with the names; like temporary nids
    unique_names = list(set(names))
    name_class_dict = {
        name: i for (name, i) in zip(unique_names, range(len(unique_names)))
    }
    classes = [name_class_dict[name] for name in names]
    files = ibs.get_annot_image_paths(aid_list)
    csv_dicts = [
        {'class': c, 'file': f, 'name': n} for (c, f, n) in zip(classes, files, names)
    ]
    _write_csv_dicts(csv_dicts, fname)
    print('PIE wrote labels csv to %s' % fname)
    return fname


# quality-control method to compare the dicts from both predict methods
# IDK if it's randomness in k-means or float errors, but the distances differ in practice by ~2e-8
@register_ibs_method
def _pie_compare_dicts(ibs, answer_dict1, answer_dict2, dist_tolerance=1e-5):

    labels1 = [entry['label'] for entry in answer_dict1]
    labels2 = [entry['label'] for entry in answer_dict2]
    agree = [lab1 == lab2 for (lab1, lab2) in zip(labels1, labels2)]
    assert all(agree), 'Label rankings differ at rank %s' % agree.index(False)
    print('Labels agree')

    distances1 = [entry['distance'] for entry in answer_dict1]
    distances2 = [entry['distance'] for entry in answer_dict2]
    diffs = [abs(d1 - d2) for (d1, d2) in zip(distances1, distances2)]
    assert max(diffs) < dist_tolerance, 'Distances diverge at rank %s' % diffs.index(
        max(diffs)
    )

    print('Distances are all within tolerance of %s' % dist_tolerance)


@register_ibs_method
def pie_accuracy(ibs, qaid, daid_list):
    daids = daid_list.copy()
    daids.remove(qaid)
    ans = ibs.pie_predict_light(qaid, daids)
    ans_names = [row['label'] for row in ans]
    ground_truth = ibs.get_annot_name_texts(qaid)
    try:
        rank = ans_names.index(ground_truth) + 1
    except ValueError:
        rank = -1
    print('rank %s' % rank)
    return rank


@register_ibs_method
def pie_mass_accuracy(ibs, aid_list, daid_list=None):
    if daid_list is None:
        daid_list = aid_list
    ranks = [ibs.pie_accuracy(aid, daid_list) for aid in aid_list]
    return ranks


@register_ibs_method
def accuracy_at_k(ibs, ranks, max_rank=10):
    counts = [ranks.count(i) for i in range(1, max_rank + 1)]
    percent_counts = [count / len(ranks) for count in counts]
    cumulative_percent = [
        sum(percent_counts[:i]) for i in range(1, len(percent_counts) + 1)
    ]
    return cumulative_percent


@register_ibs_method
def subset_with_resights(ibs, aid_list, n=3):
    names = ibs.get_annot_name_rowids(aid_list)
    name_counts = _count_dict(names)
    good_annots = [aid for aid, name in zip(aid_list, names) if name_counts[name] >= n]
    return good_annots


def _count_dict(item_list):
    from collections import defaultdict

    count_dict = defaultdict(int)
    for item in item_list:
        count_dict[item] += 1
    return dict(count_dict)


def _invert_dict(d):
    from collections import defaultdict

    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return dict(inverted)


# Careful, this returns a different ibs than you sent in
def pie_testdb_ibs():
    import wbia

    testdb_name = 'manta-test'
    try:
        ans_ibs = wbia.opendb(testdb_name)
        aids = ans_ibs.get_valid_annots()
        assert len(aids) > 3
        return ans_ibs
    except Exception:
        print("PIE testdb does not exist; creating it with PIE's example images")

    ans_ibs = wbia.opendb(testdb_name, allow_newdir=True)

    test_image_folder = os.path.join(_PLUGIN_FOLDER, 'examples/manta-demo/test')
    test_images = os.listdir(test_image_folder)
    test_images = [fname for fname in test_images if fname.lower().endswith('.png')]
    gpaths = [os.path.join(test_image_folder, fname) for fname in test_images]
    names = [fname.split('-')[0] for fname in test_images]

    gid_list = ans_ibs.add_images(gpaths)
    nid_list = ans_ibs.add_names(names)
    species = ['Mobula birostris'] * len(gid_list)
    # these images are pre-cropped aka trivial annotations
    wh_list = ans_ibs.get_image_sizes(gid_list)
    bbox_list = [[0, 0, w, h] for (w, h) in wh_list]
    ans_ibs.add_annots(
        gid_list, bbox_list=bbox_list, species_list=species, nid_list=nid_list
    )

    return ans_ibs


if __name__ == '__main__':
    import xdoctest as xdoc
    xdoc.doctest_module(__file__)
