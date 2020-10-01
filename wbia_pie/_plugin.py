# -*- coding: utf-8 -*-
import logging
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import os
import json
import shutil

try:
    import wbia

    USE_WBIA = True
except Exception:
    import ibeis as wbia

    USE_WBIA = False

if USE_WBIA:
    from wbia.control import controller_inject
    from wbia.constants import ANNOTATION_TABLE, UNKNOWN
    from wbia import dtool as dt
    import vtool as vt
else:
    from ibeis.control import controller_inject
    from ibeis.constants import ANNOTATION_TABLE, UNKNOWN
    import dtool as dt
    import vtool as vt


(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger()

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
if USE_WBIA:
    register_api = controller_inject.get_wbia_flask_api(__name__)
else:
    register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']

_PLUGIN_FOLDER = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG = os.path.join(_PLUGIN_FOLDER, 'configs/manta.json')
_DEFAULT_CONFIG_DICT = {'config_path': _DEFAULT_CONFIG}


MODEL_URLS = {
    'mobula_birostris': 'https://wildbookiarepository.azureedge.net/models/pie.manta_ray_giant.h5',
    'mobula_alfredi': 'https://wildbookiarepository.azureedge.net/models/pie.manta_ray_giant.h5',
    'manta_ray_giant': 'https://wildbookiarepository.azureedge.net/models/pie.manta_ray_giant.h5',
    'megaptera_novaeangliae': 'https://wildbookiarepository.azureedge.net/models/pie.whale_humpback.h5',
    'aetomylaeus_bovinus': 'https://wildbookiarepository.azureedge.net/models/pie.manta_ray_giant.h5',
    'whale_humpback': 'https://wildbookiarepository.azureedge.net/models/pie.whale_humpback.h5',
}


@register_ibs_method
def pie_embedding_timed(ibs, aid_list, config_path=_DEFAULT_CONFIG, use_depc=True):
    import time

    start = time.time()
    ans = ibs.pie_embedding(aid_list, config_path, None, use_depc)
    elapsed = time.time() - start
    logger.info('Computed %s embeddings in %s seconds' % (len(aid_list), elapsed))
    per_embedding = elapsed / len(aid_list)
    logger.info('average time is %s per embedding' % per_embedding)
    return ans


# note: an embedding is 256xfloat8, aka 2kb in size (using default config)
@register_ibs_method
def pie_embedding(ibs, aid_list, config_path=_DEFAULT_CONFIG, augmentation_seed=None, use_depc=True):
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
        >>> import uuid
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> aids = ibs.get_valid_aids(species='Mobula birostris')
        >>> embs_depc    = np.array(ibs.pie_embedding(aids, use_depc=True))
        >>> embs_no_depc = np.array(ibs.pie_embedding(aids, use_depc=False))
        >>> diffs = np.abs(embs_depc - embs_no_depc)
        >>> assert diffs.max() < 1e-8
        >>> # each embedding is 256 floats long so we'll just check a bit
        >>> annot_uuids = ibs.get_annot_semantic_uuids(aids)
        >>> wanted_uuid = uuid.UUID('588dc49a-9b7f-d362-1667-1f9f002cd566')
        >>> wanted_index = annot_uuids.index(wanted_uuid)
        >>> assert wanted_index is not None and wanted_index in list(range(len(aids)))
        >>> result = embs_depc[wanted_index][:20]
        >>> result_ = np.array([-0.03839333,  0.01182338,  0.02393869, -0.07164327, -0.04367629,
        >>>                     -0.00150531,  0.01324393,  0.10909598,  0.02349781,  0.08439559,
        >>>                     -0.06415793,  0.0110384 ,  0.03897202, -0.11256221,  0.00709192,
        >>>                      0.10403764,  0.00615681, -0.10405623,  0.0320793 , -0.0394897 ])
        >>> assert result.shape == result_.shape
        >>> diffs = np.abs(result - result_)
        >>> assert diffs.max() < 1e-5

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
        >>> assert compare_embs.max() < 1e-8
    """
    if use_depc:
        config = {'config_path': config_path, 'augmentation_seed': augmentation_seed}
        embeddings = ibs.depc_annot.get(
            'PieEmbedding', aid_list, 'embedding', config=config
        )
    else:
        embeddings = pie_compute_embedding(ibs, aid_list, config_path=config_path,
            augmentation_seed=augmentation_seed)
    return embeddings


class PieEmbeddingConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('config_path', _DEFAULT_CONFIG),
        ut.ParamInfo('augmentation_seed', None, hideif=None),
    ]


@register_preproc_annot(
    tablename='PieEmbedding',
    parents=[ANNOTATION_TABLE],
    colnames=['embedding'],
    coltypes=[np.ndarray],
    configclass=PieEmbeddingConfig,
    fname='pie',
    chunksize=1024,
)
@register_ibs_method
def pie_embedding_depc(depc, aid_list, config):
    ibs = depc.controller
    embs = pie_compute_embedding(ibs, aid_list, config_path=config['config_path'],
                                 augmentation_seed=config['augmentation_seed'])
    for aid, emb in zip(aid_list, embs):
        yield (np.array(emb),)


# TODO: delete the generated files in dbpath when we're done computing embeddings
@register_ibs_method
def pie_compute_embedding(
    ibs, aid_list, config_path=_DEFAULT_CONFIG, output_dir=None, prefix=None, export=False, augmentation_seed=None,
):
    preproc_dir = ibs.pie_preprocess(aid_list, config_path=config_path)
    from .compute_db import compute

    _ensure_model_exists(ibs, aid_list, config_path)

    embeddings, filepaths = compute(preproc_dir, config_path, output_dir, prefix, export, augmentation_seed)
    embeddings = fix_pie_embedding_order(ibs, embeddings, aid_list, filepaths, config_path)

    return embeddings


def _ensure_model_exists(ibs, aid_list, config_path):

    species = ibs.get_annot_species_texts(aid_list[0])
    model_url = MODEL_URLS[species]

    # get expected model location from config file. Couple lines copied from Olga's compute_db.py

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    exp_folder = os.path.join(
        _PLUGIN_FOLDER, config['train']['exp_dir'], config['train']['exp_id']
    )
    local_fpath = os.path.join(exp_folder, 'best_weights.h5')

    if os.path.isfile(local_fpath):
        return True

    # download the model and put it in the model_folder
    os.makedirs(exp_folder, exist_ok=True)
    ut.grab_file_url(model_url, download_dir=exp_folder, fname=local_fpath)
    return True


def fix_pie_embedding_order(ibs, embeddings, aid_list, filepaths, config_path):
    filepaths = [_get_parent_dir_and_fname_only(fpath) for fpath in filepaths]
    # PIE messes with extensions, so throw those away
    filepaths = [os.path.splitext(fp)[0] for fp in filepaths]

    names = ibs.get_annot_name_texts(aid_list)

    with open(config_path, 'r') as f:
        pie_config = json.load(f)

    chip_paths = ibs.pie_annot_training_chip_fpaths(aid_list, pie_config)
    fnames = [os.path.split(fname)[1] for fname in chip_paths]
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
    label_file = ibs.pie_name_csv(aid_list, fpath=label_file_path, config_path=config_path)
    chip_path = os.path.join(ibs.cachedir, 'extern_chips')
    from .preproc_db import preproc

    dbpath = preproc(chip_path, config_path, lfile=label_file, output=output_dir)
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
        logger.info('PIE preproc_dir creating output directory %s' % output_dir)
        os.makedirs(output_dir)
    logger.info('PIE preproc_dir for aids %s returning %s' % (aid_list, output_dir))
    return output_dir


# PIE's preproc and embed funcs require a .csv file linking filnames to labels (names)
@register_ibs_method
def pie_name_csv(ibs, aid_list, fpath=None, config_path=_DEFAULT_CONFIG):
    if fpath is None:
        fpath = os.path.join(_PLUGIN_FOLDER, 'examples/dev/name_map.csv')
    name_texts = ibs.get_annot_name_texts(aid_list)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    fnames = ibs.pie_annot_training_chip_fpaths(, aid_list, config)
    # only want final, file part of fpaths
    fnames = [fname.split('/')[-1] for fname in fnames]
    csv_dicts = [{'file': f, 'label': l} for (f, l) in zip(fnames, name_texts)]
    _write_csv_dicts(csv_dicts, fpath)
    logger.info('Saved PIE name file to %s' % fpath)
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
            ut.ParamInfo('query_aug_seeds', [None]),
            ut.ParamInfo('db_aug_seeds', [None]),
        ]


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    # qaid_list, daid_list = request.get_parent_rowids()
    # score_list = request.score_list
    # config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    # grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    # scores
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = daid_list_ != qaid
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = wbia.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class PieRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = False
    _tablename = 'Pie'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, overlay=True, config=None):
        depc = request.depc
        ibs = depc.controller
        chips = ibs.get_annot_chips(aid_list)
        return chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        chips = request.get_fmatch_overlayed_chip(
            [cm.qaid, aid], overlay=overlay, config=request.config
        )
        out_image = vt.stack_image_list(chips)
        return out_image

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        return cm_list

    def execute(request, *args, **kwargs):
        kwargs['use_cache'] = False
        result_list = super(PieRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [result for result in result_list if result.qaid in qaids]
        return result_list


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

    query_aug_seeds = config['query_aug_seeds']
    db_aug_seeds = config['db_aug_seeds']
    assert len(query_aug_seeds) > 0
    assert len(db_aug_seeds)    > 0

    import itertools as it
    all_aug_seed_pairs = it.product(query_aug_seeds, db_aug_seeds)
    # herein 'pie_' prefix means the var is in the original PIE match result format,
    # a list of {"label": ____, "distance": ____} dicts.
    pie_name_scores_per_aug = []

    # get name scores for every pair of augmentation seeds
    for query_aug_seed, db_aug_seed in all_aug_seed_pairs:
        pie_name_dists  = ibs.pie_predict_light(
            qaid, daids, config['config_path'], query_aug_seed, db_aug_seed,
        )
        # pie_name_dists looks like
        # [{'distance': 0.4188250198219591, 'label': '2642'},
        # {'distance': 0.46805998920189135, 'label': '1616'},
        # {'distance': 0.6673709053342388, 'label': '1131'},
        # {'distance': 0.6690026353505921, 'label': '3623'},
        # {'distance': 1.1676843213624326, 'label': '1971'},
        # {'distance': 2.377146577694036, 'label': '1804'}]
        pie_name_scores = distance_dicts_to_score_dicts(pie_name_dists)
        pie_name_scores_per_aug.append(pie_name_scores)

    avg_name_scores = average_pie_name_score_dicts(pie_name_scores_per_aug)

    aid_score_list = aid_scores_from_name_scores(ibs, avg_name_scores, daids)
    aid_score_dict = dict(zip(daids, aid_score_list))

    for daid in daid_list:
        daid_score = aid_score_dict.get(daid)
        yield (daid_score,)


def distance_to_score(distance):
    # score = 1 / (1 + distance)
    score = np.exp(-distance / 2.0)
    return score


def distance_dicts_to_score_dicts(distance_dicts, conversion_func=distance_to_score):
    score_dicts = distance_dicts.copy()
    for entry in score_dicts:
        entry['score'] = conversion_func(entry['distance'])
    return score_dicts


# list_of_name_score_dicts is a list of original-PIE-formatted name_score_dicts lists (list of list of dicts)
def average_pie_name_score_dicts(list_of_name_score_dicts):
    from collections import defaultdict
    name_to_score = defaultdict(float)

    # NOTE: this implicitly adds a score of 0 any time a name does not have a score in a given pie_name_score_dicts
    # sum the scores per name
    for pie_name_score_dicts in list_of_name_score_dicts:
        for name_score in pie_name_score_dicts:
            name = name_score['label']
            score = name_score['score']
            name_to_score[name] += score

    # divide by num augs
    n_augs = len(list_of_name_score_dicts)
    for name in name_to_score.keys():
        name_to_score[name] = name_to_score[name] / n_augs

    return name_to_score


@register_ibs_method
def aid_scores_from_name_scores(ibs, name_score_dict, daid_list):
    # import utool as ut
    # ut.embed()
    daid_name_list = list(_db_labels_for_pie(ibs, daid_list))

    name_count_dict = {name: daid_name_list.count(name)
        for name in name_score_dict.keys()}

    name_annotwise_score_dict = {name: name_score_dict[name] / name_count_dict[name]
        for name in name_score_dict.keys()}

    from collections import defaultdict
    name_annotwise_score_dict = defaultdict(float, name_annotwise_score_dict)

    # bc daid_name_list is in the same order as daid_list
    daid_scores = [name_annotwise_score_dict[name] for name in daid_name_list]
    return daid_scores


# We get a score per-name, but now we need to compute scores per-annotation. Done simply by averaging the name score over all of that name's annotations
@register_ibs_method
def aid_scores_from_name_score_dicts(ibs, name_score_dicts, daid_list):
    daid_name_list = list(_db_labels_for_pie(ibs, daid_list))
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
def pie_predict_light(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG, query_aug_seed=None, db_aug_seed=None, n_results=100):
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
        python -m wbia_pie._plugin pie_predict_light

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia_pie
        >>> import numpy as np
        >>> import uuid
        >>> ibs = wbia_pie._plugin.pie_testdb_ibs()
        >>> qannot_uuid = uuid.UUID('2a8344f2-17b1-a9b0-66d6-4f2c0e565bf1')  # name = candy
        >>> aids = ibs.get_valid_aids()
        >>> qaid = ibs.get_annot_aids_from_semantic_uuid(qannot_uuid)
        >>> daids = sorted(list(set(aids) - set([qaid])))
        >>> ibs.depc_annot.delete_property_all('PieEmbedding', aids)
        >>> pred       = ibs.pie_predict(qaid, daids)
        >>> ibs.depc_annot.delete_property_all('PieEmbedding', aids)
        >>> pred_light = ibs.pie_predict_light(qaid, daids)
        >>> expected_results = {
        >>>     'candy': 0.9821299690545198,
        >>>     'valentine': 1.139240264231686,
        >>>     'jel': 1.3051159328285846,
        >>>     'april': 1.378186756498962,
        >>> }
        >>> for value in pred + pred_light:
        >>>     label = value['label']
        >>>     diff = np.abs(value['distance'] - expected_results[label])
        >>>     print('Checking %r - diff %0.08f' % (label, diff, ))
        >>>     assert diff < 1e-6
    """
    # now get the embeddings into the shape and type PIE expects
    print("pie_predict_light called")
    db_embs = ibs.pie_embedding(daid_list, config_path, augmentation_seed=db_aug_seed)
    query_emb = ibs.pie_embedding([qaid], config_path, augmentation_seed=query_aug_seed)
    db_labels = _db_labels_for_pie(ibs, daid_list)

    from .predict import pred_light

    nearest_neighbors_cache_path = os.path.join(ibs.cachedir, 'pie_neighbors')
    ut.ensuredir(nearest_neighbors_cache_path)

    ans = pred_light(query_emb, db_embs, db_labels, config_path, n_results,
                     nearest_neighbors_cache_path=nearest_neighbors_cache_path)
    return ans


def _db_labels_for_pie(ibs, daid_list):
    db_labels = ibs.get_annot_name_texts(daid_list)
    db_auuids = ibs.get_annot_semantic_uuids(daid_list)
    # later we must know which db_labels are for single auuids, hence prefix
    db_auuids = [UNKNOWN + str(auuid) for auuid in db_auuids]
    db_labels = [
        lab if lab is not UNKNOWN else auuid for lab, auuid in zip(db_labels, db_auuids)
    ]
    db_labels = np.array(db_labels)
    return db_labels


@register_ibs_method
def pie_predict_light_2(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG):
    db_embs = np.array(ibs.pie_embedding(daid_list))
    db_labels = np.array(ibs.get_annot_name_texts(daid_list))
    # todo: cache this
    query_emb = ibs.pie_compute_embedding([qaid])
    from .predict import pred_light

    nearest_neighbors_cache_path = os.path.join(ibs.cachedir, 'pie_neighbors')
    ut.ensuredir(nearest_neighbors_cache_path)

    ans = pred_light(query_emb, db_embs, db_labels, config_path,
                     nearest_neighbors_cache_path=nearest_neighbors_cache_path)
    return ans


# PIE's predict.predict function is more of a command-line program than a package function. It works off of .csv files, .json files, and images stored in hierachical folders, and does all sorts of preprocessing assuming the qaid embedding hasn't been computed before (so ofc, it computes that). This method is basically deprecated and was a proof-of-concept/MVP, and pie_predict_light should be used in practice. Kept around so we can verify that pie_predict_light produces the same results as the released predict function.
@register_ibs_method
def pie_predict(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG, display=False):
    config = ibs.pie_predict_prepare_config(daid_list, _DEFAULT_CONFIG)
    impath = ibs.get_annot_image_paths(qaid)

    from .predict import predict

    nearest_neighbors_cache_path = os.path.join(ibs.cachedir, 'pie_neighbors')
    ut.ensuredir(nearest_neighbors_cache_path)

    ans = predict(impath, config, config_path, display,
                  nearest_neighbors_cache_path=nearest_neighbors_cache_path)
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
        logger.info('PIE embeddings_dir creating output directory %s' % output_dir)
        os.makedirs(output_dir)
    return output_dir


def _write_embeddings_csv(embeddings, fname, base_config_path=_DEFAULT_CONFIG):
    ncols = len(embeddings[0])
    header = ['emb_' + str(i) for i in range(ncols)]
    header = ','.join(header)
    np.savetxt(fname, embeddings, delimiter=',', newline='\n', header=header)
    logger.info('PIE wrote embeddings csv to %s' % fname)
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
    logger.info('PIE wrote labels csv to %s' % fname)
    return fname


# quality-control method to compare the dicts from both predict methods
# IDK if it's randomness in k-means or float errors, but the distances differ in practice by ~2e-8
@register_ibs_method
def _pie_compare_dicts(ibs, answer_dict1, answer_dict2, dist_tolerance=1e-5):

    labels1 = [entry['label'] for entry in answer_dict1]
    labels2 = [entry['label'] for entry in answer_dict2]
    agree = [lab1 == lab2 for (lab1, lab2) in zip(labels1, labels2)]
    assert all(agree), 'Label rankings differ at rank %s' % agree.index(False)
    logger.info('Labels agree')

    distances1 = [entry['distance'] for entry in answer_dict1]
    distances2 = [entry['distance'] for entry in answer_dict2]
    diffs = [abs(d1 - d2) for (d1, d2) in zip(distances1, distances2)]
    assert max(diffs) < dist_tolerance, 'Distances diverge at rank %s' % diffs.index(
        max(diffs)
    )

    logger.info('Distances are all within tolerance of %s' % dist_tolerance)



# Scripts `train.py` and `evaluate.py` have been used to train and evaluate a network configuration specified in a config file.
# python train.py -c configs/manta.json


@register_ibs_method
def pie_training(ibs, training_aids, base_config_path=_DEFAULT_CONFIG):
    # TODO: do we change the config file?
    # preproc_dir = ibs.pie_preprocess(training_aids, base_config_path)

    with open(base_config_path, 'r') as f:
        config = json.load(f)

    _prepare_training_images(ibs, training_aids, config)

    from .train import train
    import datetime
    print('%s: about to train' % datetime.datetime.now())
    ans = train(config)
    print('%s: done training' % datetime.datetime.now())
    return ans


@register_ibs_method
def pie_aerial_training(ibs, base_config_path='configs/rw-aerial.json'):

    # skipping bc should be one-time
    # ibs.pie_prepare_rw_aerials(base_config_path)
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(_PLUGIN_FOLDER, base_config_path)
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    from .train import train
    import datetime
    print("%s: about to train" % datetime.datetime.now())
    ans = train(config)
    print("%s: done training" % datetime.datetime.now())
    return ans


@register_ibs_method
def pie_evaluate(ibs, config_path=_DEFAULT_CONFIG):
    # TODO: do we change the config file?
    # preproc_dir = ibs.pie_preprocess(training_aids, base_config_path)
    from .evaluate import evaluate
    import datetime
    print('%s: about to evaluate' % datetime.datetime.now())
    ans = evaluate(config_path)
    print('%s: done evaluating' % datetime.datetime.now())
    return ans


def _prepare_training_images(ibs, aid_list, pie_config):
    #  prepare training images directory
    target_dir = pie_config['data']['train_image_folder']
    # if target_dir is a relative path, make it absolute relative this plugin directory
    if not os.path.isabs(target_dir):
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(plugin_folder, target_dir)

    # copy resized annot chips into name-based subfolders
    names = ibs.get_annot_name_texts(aid_list)
    chip_paths = ibs.pie_annot_training_chip_fpaths(aid_list, pie_config)
    for (aid, name, fpath) in zip(aid_list, names, chip_paths):
        name_dir = os.path.join(target_dir, name)
        os.makedirs(name_dir, exist_ok=True)
        shutil.copy(fpath, name_dir)


@register_ibs_method
def pie_annot_training_chip_fpaths(ibs, aid_list, pie_config):
    width  = int(pie_config['model']['input_width'])
    height = int(pie_config['model']['input_height'])

    chip_config = {
        'dim_size': (width, height),
        'resize_dim': 'wh',
        'ext': '.png',  ## example images are .png
    }

    fpaths = ibs.get_annot_chip_fpath(aid_list, ensure=True, config2_=chip_config)
    return fpaths


@register_ibs_method
def pie_rw_subset(ibs, aid_list, min_score=0.7, vpoint='left'):
    names = ibs.get_annot_name_texts(aid_list)
    views = ibs.get_annot_viewpoints(aid_list)
    confs = ibs.get_annot_detect_confidence(aid_list)
    # we want all the annots that have viewpoint not 'up' or None and do have a name (ie name_text!='____')

    annot_subset = [aid for (aid, name, view, conf) in zip(aid_list, names, views, confs) if
                    name != '____' and
                    view == vpoint and
                    conf >= min_score
                    ]

    # enforce range of sightings per name
    annot_subset = ibs.subset_with_resights_range(annot_subset, 3, 1000)
    names_subset = ibs.get_annot_name_texts(annot_subset)

    # just wanna compute some name statistics
    name_counts = _count_dict(names_subset)
    name_hist = [name_counts[name] for name in name_counts.keys()]
    name_hist = _count_dict(name_hist)
    # print("name-sightings histogram:")
    # print(name_hist)
    num_names = len(set(names_subset))
    annots_per_name = len(annot_subset) / num_names
    # print("%s and up =============================" % min_score)
    print('%s viewpoint, %s min_score: %s annots per name (%s names, %s annots)' % (vpoint, min_score,
        '{:6.1f}'.format(annots_per_name), '{0:3}'.format(num_names),
        '{0:3}'.format(len(annot_subset))))
    return annot_subset


def csv_to_dicts(fname):
    import csv
    dicts = []
    with open(fname, 'r') as data:
        for line in csv.DictReader(data):
            dicts.append(line)
    return dicts


@register_ibs_method
def pie_rw_subset_2(ibs, aid_list, min_sights=3, side='L'):
    fname = os.path.join(_PLUGIN_FOLDER, 'rw/photosIDMapHead_L_R.csv')
    csv_rows = csv_to_dicts(fname)
    narw_im_names = [row['Encounter.MediaAsset'] for row in csv_rows]
    narw_views = [row['Concat_ViewDirectionCode'] for row in csv_rows]
    narw_image_to_viewcode = {im_name: view for (im_name, view) in zip(narw_im_names, narw_views)}
    from collections import defaultdict  # so we can throw anything at image_to_viewcode without key errors
    narw_image_to_viewcode = defaultdict(str, narw_image_to_viewcode)

    ib_views = ibs.get_annot_viewpoints(aid_list)
    ib_im_names = ibs.get_annot_image_names(aid_list)

    # grab annots where viewpoints agree in both sources
    # side_view and side are just two diff labels for same thing, but in two diff places
    side_view = 'left' if side == 'L' else 'right'
    good_annots = [aid for aid, view, im_name in zip(aid_list, ib_views, ib_im_names)
                   if view == side_view and narw_image_to_viewcode[im_name] == side]
    good_annots = ibs.subset_with_resights(good_annots, n=min_sights)
    # just wanna compute some name statistics
    good_names = ibs.get_annot_names(good_annots)
    name_counts = _count_dict(good_names)
    name_hist = [name_counts[name] for name in name_counts.keys()]
    name_hist = _count_dict(name_hist)
    print('name hist:')
    print(name_hist)
    num_names = len(set(good_names))
    annots_per_name = len(good_annots) / num_names
    # print("%s and up =============================" % min_score)
    print('%s annots on %s side per name (%s names, %s annots)' % (
        '{:6.1f}'.format(annots_per_name), side,
        '{0:3}'.format(num_names), '{0:3}'.format(len(good_annots))))
    return good_annots


@register_ibs_method
def pie_rw_subset_3_drew(ibs, aid_list, min_sights=6, max_sights=20, side='L'):
    fname = os.path.join(_PLUGIN_FOLDER, 'rw/photosIDMapHead_L_R.csv')
    csv_rows = csv_to_dicts(fname)
    narw_im_names = [row['Encounter.MediaAsset'] for row in csv_rows]
    narw_views = [row['Concat_ViewDirectionCode'] for row in csv_rows]
    narw_image_to_viewcode = {im_name: view for (im_name, view) in zip(narw_im_names, narw_views)}
    from collections import defaultdict  # so we can throw anything at image_to_viewcode without key errors
    narw_image_to_viewcode = defaultdict(str, narw_image_to_viewcode)
    ib_im_names = ibs.get_annot_image_names(aid_list)
    good_annots = [aid for aid, im_name in zip(aid_list, ib_im_names)
                   if narw_image_to_viewcode[im_name] == side]

    # because we trust these viewpoints
    good_annots = filter_out_viewpoints(ibs, good_annots, bad_views=['up','right','front'])
    good_annots = size_filter_aids(ibs, good_annots, min_width=448, min_height=224)
    good_annots = only_single_annot_images(ibs, good_annots)
    good_annots = ibs.subset_with_resights_range(good_annots, min_sights, max_sights)
    # just wanna compute some name statistics
    good_names = ibs.get_annot_names(good_annots)
    name_counts = _count_dict(good_names)
    name_hist = [name_counts[name] for name in name_counts.keys()]
    name_hist = _count_dict(name_hist)
    print('name hist:')
    print(name_hist)
    num_names = len(set(good_names))
    annots_per_name = len(good_annots) / num_names
    # print("%s and up =============================" % min_score)
    print('%s annots on %s side per name (%s names, %s annots)' % (
        '{:6.1f}'.format(annots_per_name), side,
        '{0:3}'.format(num_names), '{0:3}'.format(len(good_annots))))
    return good_annots


@register_ibs_method
def pie_prepare_rw_aerials(ibs, config_path='configs/rw-aerial.json', min=3):
    fname = '/home/wildme/code/ibeis-deepsense-module/retraining/kaggle-passports.csv'
    csv_rows = csv_to_dicts(fname)


    if not os.path.isabs(config_path):
        config_path = os.path.join(_PLUGIN_FOLDER, config_path)
    with open(config_path, 'r') as f:
        pie_config = json.load(f)

    target_dir = pie_config['data']['train_image_folder']
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(_PLUGIN_FOLDER, target_dir)

    names  = [row['whaleID'] for row in csv_rows]
    fpaths = [row['Image']   for row in csv_rows]

    good_fpaths, good_names = subset_with_resights_helper(names, fpaths, min)

    # just wanna compute some name statistics
    name_counts = _count_dict(good_names)
    name_hist = [name_counts[name] for name in name_counts.keys()]
    name_hist = _count_dict(name_hist)
    n_names = len(set(good_names))
    print("%s names, %s sights, %s per" % (n_names, len(good_fpaths), len(good_fpaths)/n_names))
    print("name-sightings histogram:")
    print(name_hist)

    print("about to copy %s files into %s" %(len(good_fpaths), target_dir))
    for (name, fpath) in zip(good_names, good_fpaths):
        name_dir = os.path.join(target_dir, name)
        os.makedirs(name_dir, exist_ok=True)
        shutil.copy(fpath, name_dir)


def only_single_annot_images(ibs, aid_list):
    im_names = ibs.get_annot_image_names(aid_list)
    name_counts = _count_dict(im_names)
    good_annots = [aid for (aid, im_name) in zip(aid_list, im_names)
        if name_counts[im_name] == 1]
    return good_annots


def size_filter_aids(ibs, aid_list, min_width=448, min_height=224):
    bboxes = ibs.get_annot_bboxes(aid_list)
    widths = [bbox[2] for bbox in bboxes]
    heights = [bbox[3] for bbox in bboxes]

    good_aids = [aid for (aid, w, h) in zip(aid_list, widths, heights) if
                 w >= min_width and h >= min_height]
    return good_aids


def filter_out_viewpoints(ibs, aid_list, bad_views=['up','right','front']):
    bad_views = set(bad_views)
    views = ibs.get_annot_viewpoints(aid_list)
    good_annots = [aid for (aid, view) in zip(aid_list, views)
                   if view not in bad_views]
    return good_annots


def gradient_magnitude(arguments):
    import numpy as np
    import cv2

    image_filepath, = arguments

    image = cv2.imread(image_filepath)
    image = image.astype(np.float32)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)

    result = {
        'sum': np.sum(magnitude),
        'mean': np.mean(magnitude),
        'max': np.max(magnitude),
    }
    return result


def background_mask_points(arguments):
    import cv2
    image_filepath, = arguments

    margin = 8
    k = 13
    image = cv2.imread(image_filepath, 0)
    image = image.astype(np.float32)
    image[image < 32] = 0
    # image[image > 0] = 255
    kernel = np.ones((k, k), np.uint8)
    image[:margin, :] = 0
    image[-margin:, :] = 0
    image[:, :margin] = 0
    image[:, -margin:] = 0
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.blur(image, (k, k))
    image = np.around(image)
    image[image < 32] = 0
    image[image > 0] = 255

    try:
        image_x = np.sum(image, axis=0)
        for index in range(len(image_x)):
            if image_x[index] > 0:
                start_x = index
                break
        start_y = int(np.around(np.mean(ut.flatten(np.argwhere(image[:, start_x] > 0)))))
    except:
        return None, None
    try:
        image_y = np.sum(image, axis=1)
        for index in range(len(image_y)):
            if image_y[index] > 0:
                end_y = index
                break
        end_x = int(np.around(np.mean(ut.flatten(np.argwhere(image[end_y, :] > 0)))))
    except:
        return None, None

    distance = np.sqrt((end_y - start_y) ** 2.0 + (end_x - start_x) ** 2.0)
    radians = np.arctan2(end_y - start_y, end_x - start_x)
    angle = np.rad2deg(radians)

    return distance, angle


def value_deltas(values):
        assert None not in values
        assert -1 not in values
        previous = 0
        delta_list = []
        for value in values + [None]:
            if value is None:
                break
            else:
                try:
                    delta = value - previous
                except:
                    delta = 0
            assert delta >= 0
            delta_list.append(delta)
            previous = value
        assert len(delta_list) == len(values)
        delta_list = np.array(delta_list)
        return delta_list


@register_ibs_method
def pie_rw_subset_3_jrp(ibs, aid_list=None, min_sights=2, max_sights=np.inf, side='L'):
    """
    Example:
    >>> import concurrent.futures
    >>> import numpy as np
    >>> import random
    >>> import tqdm
    >>> import cv2
    >>> import os
    >>> globals().update(locals())
    >>> from wbia_pie._plugin import *
    >>> aid_list=None
    >>> min_sights=2
    >>> max_sights=np.inf
    >>> side='L'
    """
    import concurrent.futures
    import numpy as np
    import random
    import tqdm
    import cv2
    import os

    random.seed(777)

    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    side = side.upper()
    assert side in ['L', 'R']
    desired_view = 'left' if side == 'L' else 'right'

    MIN_ANNOTS_PER_NAME = min_sights
    MAX_ANNOTS_PER_NAME = max_sights
    CHIP_HEIGHT = 224
    CHIP_WIDTH = 448
    MIN_HEIGHT = int(np.floor(CHIP_HEIGHT * 0.75))
    MIN_WIDTH = int(np.floor(CHIP_WIDTH * 0.75))
    MIN_GID_DELTA = 150

    # _plugin_folder = '/data/jason.parham/code/wbia-plugin-pie/wbia_pie/'
    _plugin_folder = _PLUGIN_FOLDER

    # fname = os.path.join(_plugin_folder, 'rw/photosIDMapHead_L_R.csv')
    fname = os.path.join(_plugin_folder, 'rw/photosIDMapHead_L_R.new.csv')
    csv_rows = csv_to_dicts(fname)

    image_ids   = ut.take_column(csv_rows, 'CatalogImageId')
    image_names = ut.take_column(csv_rows, 'Encounter.MediaAsset')
    directions0  = ut.take_column(csv_rows, 'Concat_ViewDirectionCode')
    directions1 = ut.take_column(csv_rows, 'Concat_ViewDirectionCode1')
    directions2 = ut.take_column(csv_rows, 'Concat_ViewDirectionCode2')
    directions3 = ut.take_column(csv_rows, 'Concat_ViewDirectionCode3')
    name_texts  = ut.take_column(csv_rows, 'MarkedIndividual.individualID')
    years       = ut.take_column(csv_rows, 'Encounter.year')
    primaries   = ut.take_column(csv_rows, 'PrimaryImage')
    deaths      = ut.take_column(csv_rows, 'DeathYear')

    image_ids_ = [os.path.splitext(image_name)[0] for image_name in image_names]
    assert all([image_id == image_id_ for image_id, image_id_ in zip(image_ids, image_ids_)])
    directions = list(zip(directions0, directions1, directions2, directions3))
    directions = [
        list(set(direction) - set(['']))
        for direction in directions
    ]
    len_list = list(map(len, directions))
    assert len_list.count(1) == len(len_list)
    assert set(len_list) == set([1])
    directions = ut.flatten(directions)
    assert set(directions) == set(['L', 'R'])

    years = list(map(int, years))
    years = np.array(years)
    min_year = int(np.mean(years) - np.std(years))

    primaries = list(map(bool, primaries))
    primaries = np.array(primaries)

    deaths = [None if len(death) == 0 else int(death) for death in deaths]

    image_ids = list(map(int, image_ids))
    name_texts = list(map(int, name_texts))

    assert len(image_ids) == len(name_texts)
    assert len(image_ids) == len(directions)

    zipped = zip(image_ids, name_texts, directions, years, primaries, deaths)
    data_dict = {
        image_id: {
            'name': str(name).lower(),
            'view': ('left' if direction == 'L' else 'right').lower(),
            'year': year,
            'prime': primary,
            'dead': death,
        }
        for image_id, name, direction, year, primary, death in zipped
    }

    gid_list = ibs.get_valid_gids()
    image_uri_original_list = ibs.get_image_uris_original(gid_list)

    gid_dict = {}
    seen_image_id_set = set([])
    for gid, image_uri_original in zip(gid_list, image_uri_original_list):
        image_filename_original = os.path.split(image_uri_original)[1]
        image_id_ = os.path.splitext(image_filename_original)[0]
        image_id_ = int(image_id_)

        if image_id_ in seen_image_id_set:
            print('Already seen image_id = %r' % (image_id_, ))
        data = data_dict.get(image_id_, None)
        if data is None:
            continue
        gid_dict[gid] = data
        seen_image_id_set.add(image_id_)

    candidate_gid_list = []
    candidate_name_list = []
    candidate_primary_list = []
    for gid in gid_dict:
        data = gid_dict[gid]
        name = data.get('name')
        view = data.get('view')
        year = data.get('year')
        dead = data.get('dead')
        prime = data.get('prime')
        if view != desired_view:
            continue
        if year < min_year:
            continue
        if dead is not None:
            continue
        candidate_gid_list.append(gid)
        candidate_name_list.append(name)
        candidate_primary_list.append(prime)

    assert len(candidate_gid_list) == len(set(candidate_gid_list))
    assert len(candidate_gid_list) == len(candidate_name_list)
    assert len(candidate_gid_list) == len(candidate_primary_list)

    candidate_aids_list = ibs.get_image_aids(candidate_gid_list)
    # len_list = list(map(len, candidate_aids_list))

    labeler_config = {
        'labeler_algo': 'densenet',
        'labeler_weight_filepath': 'right_whale_v0',
    }
    candidate_aid_list = ut.flatten(candidate_aids_list)
    labeler_species_list = ibs.depc_annot.get_property('labeler', candidate_aid_list, 'species', config=labeler_config)
    labeler_viewpoint_list = ibs.depc_annot.get_property('labeler', candidate_aid_list, 'viewpoint', config=labeler_config)
    zipped = zip(candidate_aid_list, labeler_species_list, labeler_viewpoint_list)
    labeler_dict = {
        candidate_aid: {
            'species': labeler_species,
            'viewpoint': labeler_viewpoint,
        }
        for candidate_aid, labeler_species, labeler_viewpoint in zipped
    }

    chip_config = {
        'dim_size': (CHIP_WIDTH, CHIP_HEIGHT),
        'resize_dim': 'wh',
        'ext': '.png',
    }
    candidate_chip_filepath_list = ibs.get_annot_chip_fpath(candidate_aid_list, ensure=True, config2_=chip_config)

    arguments_list = list(zip(candidate_chip_filepath_list))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        candidate_gradient_result_list = list(
            tqdm.tqdm(
                executor.map(gradient_magnitude, arguments_list),
                total=len(arguments_list),
            )
        )
    candidate_gradient_dict = dict(zip(candidate_aid_list, candidate_gradient_result_list))

    candidate_gradient_mean_list = ut.take_column(candidate_gradient_result_list, 'mean')
    candidate_gradient_mean_list = np.array(candidate_gradient_mean_list)
    candidate_gradient_mean = np.mean(candidate_gradient_mean_list)
    candidate_gradient_std = np.std(candidate_gradient_mean_list)
    candidate_gradient_mean_min = int(np.floor(candidate_gradient_mean - 1.0 * candidate_gradient_std))
    candidate_gradient_mean_max = int(np.ceil(candidate_gradient_mean + 2.0 * candidate_gradient_std))

    flag_list = np.logical_or(candidate_gradient_mean_list < candidate_gradient_mean_min, candidate_gradient_mean_list > candidate_gradient_mean_max)
    print('Ignoring %d candidates due to gradient' % (sum(flag_list), ))

    current_species_list = ibs.get_annot_species(candidate_aid_list)
    ibs.set_annot_species(candidate_aid_list, ['right_whale_head'] * len(candidate_aid_list))
    candidate_bg_filepath_list = ibs.get_annot_probchip_fpath(candidate_aid_list)
    ibs.set_annot_species(candidate_aid_list, current_species_list)

    arguments_list = list(zip(candidate_bg_filepath_list))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        candidate_mask_point_result_list = list(
            tqdm.tqdm(
                executor.map(background_mask_points, arguments_list),
                total=len(arguments_list),
            )
        )
    candidate_mask_point_dict = dict(zip(candidate_aid_list, candidate_mask_point_result_list))

    candidate_primary_aid_set = set([])
    candidate_name_dict = {}
    zipped = zip(candidate_gid_list, candidate_aids_list, candidate_name_list, candidate_primary_list)
    for candidate_gid, candidate_aid_list, candidate_name, candidate_primary in zipped:
        if len(candidate_aid_list) == 1:
            candidate_aid = candidate_aid_list[0]

            labeler_data = labeler_dict.get(candidate_aid, None)
            if labeler_data is None:
                continue
            labeler_species = labeler_data.get('species')
            labeler_viewpoint = labeler_data.get('viewpoint')
            if labeler_species not in ['right_whale+head']:
                continue
            if labeler_viewpoint not in ['left']:
                continue

            if not candidate_primary:
                candidate_bbox = ibs.get_annot_bboxes(candidate_aid)
                xtl, ytl, w, h = candidate_bbox
                if w < MIN_WIDTH:
                    continue
                if h < MIN_HEIGHT:
                    continue

                gradient_data = candidate_gradient_dict.get(candidate_aid, None)
                if gradient_data is None:
                    continue
                gradient_mean = gradient_data.get('mean')
                if gradient_mean < candidate_gradient_mean_min:
                    continue
                if gradient_mean > candidate_gradient_mean_max:
                    continue

                mask_point_data = candidate_mask_point_dict.get(candidate_aid, None)
                if mask_point_data is None:
                    continue
                point_distance = mask_point_data[0]
                point_angle = mask_point_data[1]
                if point_distance is None or point_angle is None:
                    continue
                if point_distance < CHIP_WIDTH // 4:
                    continue
                if point_angle > -10.0:
                    continue
                if point_angle < -30.0:
                    continue
            else:
                candidate_primary_aid_set.add(candidate_aid)

            if candidate_name not in candidate_name_dict:
                candidate_name_dict[candidate_name] = []
            candidate_name_dict[candidate_name].append(candidate_aid)

    total_candidate_aids = len(ut.flatten(candidate_name_dict.values()))
    total_candidate_nids = len(candidate_name_dict)
    print('Pre-limited candidates: %d aids for %d names' % (total_candidate_aids, total_candidate_nids, ))

    for name in candidate_name_dict:
        rowid_list = candidate_name_dict[name]

        primary_rowid_list = list(set(rowid_list) & candidate_primary_aid_set)
        rowid_list = list(set(rowid_list) - candidate_primary_aid_set)

        value_list = ibs.get_annot_gids(rowid_list)

        values = sorted(zip(value_list, rowid_list))
        value_list = ut.take_column(values, 0)
        rowid_list = ut.take_column(values, 1)

        delta_list = value_deltas(value_list)
        candidate_list = delta_list < MIN_GID_DELTA
        while True in candidate_list:
            candidate_index = np.argmin(delta_list)
            # print('Popping index %d' % (candidate_index, ))
            value_list.pop(candidate_index)
            rowid_list.pop(candidate_index)
            delta_list = value_deltas(value_list)
            candidate_list = delta_list < MIN_GID_DELTA

        candidate_name_dict[name] = rowid_list + primary_rowid_list

    total_candidate_aids = len(ut.flatten(candidate_name_dict.values()))
    total_candidate_nids = len(candidate_name_dict)
    print('Pre-limited* candidates: %d aids for %d names' % (total_candidate_aids, total_candidate_nids, ))

    final_aid_list = []
    final_name_list = []
    for name in candidate_name_dict:
        candidate_name_aid_list = candidate_name_dict[name]
        if len(candidate_name_aid_list) < MIN_ANNOTS_PER_NAME:
            continue
        primary_candidate_name_aid_list = list(set(candidate_name_aid_list) & candidate_primary_aid_set)
        if len(primary_candidate_name_aid_list) > 0:
            if MAX_ANNOTS_PER_NAME < len(candidate_name_aid_list):
                raise NotImplementedError()
        random.shuffle(candidate_name_aid_list)
        keep_annots = min(len(candidate_name_aid_list), MAX_ANNOTS_PER_NAME)
        candidate_name_aid_list_ = candidate_name_aid_list[:keep_annots]
        candidate_name_aid_list_ = sorted(candidate_name_aid_list_)
        candidate_name_gid_list_ = ibs.get_annot_gids(candidate_name_aid_list_)
        assert len(candidate_name_gid_list_) == len(set(candidate_name_gid_list_))

        final_aid_list += candidate_name_aid_list_
        final_name_list += [name] * len(candidate_name_aid_list_)

    assert len(final_aid_list) == len(final_name_list)
    assert len(final_aid_list) == len(set(final_aid_list))
    final_gid_list = ibs.get_annot_gids(final_aid_list)
    assert len(final_gid_list) == len(set(final_gid_list))

    print('Post-limited candidates: %d aids for %d names' % (len(final_aid_list), len(set(final_name_list)), ))

    # existing_viewpoint_list = ibs.get_annot_viewpoints(final_aid_list)
    existing_name_list = ibs.get_annot_names(final_aid_list)

    flag_list = [
        final_name == existing_name
        for final_name, existing_name in zip(final_name_list, existing_name_list)
    ]
    assert sum(flag_list) == len(flag_list)

    imageset_name = 'Jason Subset'
    imageset_rowid, = ibs.get_imageset_imgsetids_from_text([imageset_name])
    gid_list = ibs.get_valid_gids()
    ibs.unrelate_images_and_imagesets(gid_list, [imageset_rowid] * len(gid_list))
    ibs.set_image_imagesettext(final_gid_list, [imageset_name] * len(final_gid_list))

    final_aid_list = list(set(final_aid_list) & set(aid_list))
    final_aid_list = sorted(final_aid_list)

    final_name_list = ibs.get_annot_names(final_aid_list)
    final_chip_filepath_list = ibs.get_annot_chip_fpath(final_aid_list, ensure=True, config2_=chip_config)

    current_species_list = ibs.get_annot_species(final_aid_list)
    ibs.set_annot_species(final_aid_list, ['right_whale_head'] * len(final_aid_list))
    final_bg_filepath_list = ibs.get_annot_probchip_fpath(final_aid_list)
    ibs.set_annot_species(final_aid_list, current_species_list)

    output_path = os.path.join(ibs.dbdir, 'rw-training-dump-jrp-examples')
    ut.delete(output_path)
    ut.ensuredir(output_path)

    with open(os.path.join(output_path, 'names.csv'), 'w') as output_file:
        final_image_uri_original_list = ibs.get_image_uris_original(final_gid_list)
        for final_image_uri_original, final_name in zip(final_image_uri_original_list, final_name_list):
            final_filename_original = os.path.splitext(os.path.split(final_image_uri_original)[1])[0]
            output_file.write('%s,%s\n' % (final_filename_original, final_name, ))

    zipped = list(zip(final_aid_list, final_name_list, final_chip_filepath_list, final_bg_filepath_list))
    # zipped = zipped[:100]
    for final_aid, final_name, final_chip_filepath, final_bg_filepath in tqdm.tqdm(zipped):
        output_name_path = os.path.join(output_path, final_name)
        ut.ensuredir(output_name_path)
        output_chip_filename_path = os.path.join(output_name_path, '%d.jpg' % (final_aid, ))
        image = cv2.imread(final_chip_filepath)
        cv2.imwrite(output_chip_filename_path, image)

        # ut.copy(final_chip_filepath, output_chip_filename_path, verbose=False)
        # output_bg_filename_path = os.path.join(output_name_path, '%d.mask.png' % (final_aid, ))
        # ut.copy(final_bg_filepath, output_bg_filename_path, verbose=False)

        # margin = 8
        # k = 13
        # input_filepath = output_bg_filename_path
        # output_filepath = input_filepath.replace('.mask.png', '.mask.v0.png')
        # image = cv2.imread(input_filepath, 0)
        # image = image.astype(np.float32)
        # image[image < 32] = 0
        # # image[image > 0] = 255
        # kernel = np.ones((k, k), np.uint8)
        # image[:margin, :] = 0
        # image[-margin:, :] = 0
        # image[:, :margin] = 0
        # image[:, -margin:] = 0
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # image = cv2.blur(image, (k, k))
        # image = np.around(image)
        # image[image < 32] = 0
        # image[image > 0] = 255
        # mask = image.astype(np.uint8)
        # cv2.imwrite(output_filepath, mask)

        # input_filepath = output_chip_filename_path
        # output_filepath = input_filepath.replace('.png', '.composite.png')
        # image = cv2.imread(input_filepath)
        # image = image.astype(np.float32)

        # mask = mask.astype(np.float32)
        # mask /= 255.0
        # mask = cv2.resize(mask, image.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
        # mask_ = cv2.merge((mask, mask, mask))
        # composite = image * mask_

        # try:
        #     mask_x = np.sum(mask, axis=0)
        #     for index in range(len(mask_x)):
        #         if mask_x[index] > 0:
        #             start_x = index
        #             break
        #     start_y = int(np.around(np.mean(ut.flatten(np.argwhere(mask[:, start_x] > 0)))))
        #     cv2.circle(composite, (start_x, start_y), 10, (0, 255, 0))
        # except:
        #     pass
        # try:
        #     mask_y = np.sum(mask, axis=1)
        #     for index in range(len(mask_y)):
        #         if mask_y[index] > 0:
        #             end_y = index
        #             break
        #     end_x = int(np.around(np.mean(ut.flatten(np.argwhere(mask[end_y, :] > 0)))))

        #     cv2.circle(composite, (end_x, end_y), 10, (0, 0, 255))
        # except:
        #     pass

        # cv2.imwrite(output_filepath, composite)

        # input_filepath = output_chip_filename_path
        # output_filepath = input_filepath.replace('.png', '.rotated.png')
        # image = cv2.imread(input_filepath)

        # radians = np.arctan2(end_y - start_y, end_x - start_x)
        # angle = np.rad2deg(radians)
        # # print(angle)
        # h, w = image.shape[:2]
        # center = (w / 2, h / 2)
        # # Perform the rotation
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # rotated = cv2.warpAffine(image, M, (w, h))

        # cv2.imwrite(output_filepath, rotated)

    print('Rendered to %r' % (output_path, ))

    return final_aid_list


def pie_apply_names(ibs, aid_list):
    names = ibs.get_annot_name_texts(aid_list)
    nameless_aids = [aid for aid,name in zip(aid_list, names) if name == '____']

    nameless_fnames = ibs.get_annot_image_paths(nameless_aids)
    nameless_fnames = [os.path.split(fn)[1] for fn in nameless_fnames]

    # load metadata file
    import csv
    csv_path = '/home/wildme/tmp/photosIDMap.csv'
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        csv_dicts = [{k: v for k, v in row.items()} for row in csv.DictReader(f)]

    fname_to_name = {row['ImageFile']: row['Name'] for row in csv_dicts}
    fnames_with_names = set(fname_to_name.keys())
    nameless_names = [fname_to_name[fn]
                      if fn in fnames_with_names
                      else '____'
                      for fn in nameless_fnames
                      ]
    ibs.set_annot_name_texts(nameless_aids, nameless_names)


def _pie_accuracy(ibs, qaid, daid_list, config_path=_DEFAULT_CONFIG):
    daids = daid_list.copy()
    daids.remove(qaid)
    ans = ibs.pie_predict_light(qaid, daids, config_path=config_path)
    ans_names = [row['label'] for row in ans]
    ground_truth = ibs.get_annot_name_texts(qaid)
    try:
        rank = ans_names.index(ground_truth) + 1
    except ValueError:
        rank = -1
    logger.info('rank %s' % rank)
    return rank


def _count_dict(item_list):
    from collections import defaultdict, OrderedDict
    count_dict = defaultdict(int)
    for item in item_list:
        count_dict[item] += 1
    count_dict = OrderedDict(sorted(count_dict.items()))
    return dict(count_dict)


@register_ibs_method
def pie_accuracy(ibs, aid_list, daid_list=None, config_path=_DEFAULT_CONFIG):
    if daid_list is None:
        daid_list = aid_list
    ranks = [_pie_accuracy(ibs, aid, daid_list, config_path) for aid in aid_list]
    accs = ibs.accuracy_at_k(ranks)
    return accs


@register_ibs_method
def accuracy_at_k(ibs, ranks, max_rank=12):
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


def subset_with_resights_helper(names, imgs, n=3):
    name_counts = _count_dict(names)
    good_pairs = [(im,name) for im, name in zip(imgs, names) if name_counts[name] >= n]
    good_imgs  = [pair[0] for pair in good_pairs]
    good_names = [pair[1] for pair in good_pairs]
    return good_imgs, good_names


@register_ibs_method
def subset_with_resights_range(ibs, aid_list, min_sights=3, max_sights=100):
    name_to_aids = _name_dict(ibs, aid_list)
    final_aids = []
    import random
    random.seed(777)

    for name, aids in name_to_aids.items():
        if len(aids) < min_sights:
            continue
        elif len(aids) <= max_sights:
            final_aids += aids
        else:
            final_aids += random.sample(aids, max_sights)

    final_aids.sort()
    return final_aids


def _name_dict(ibs, aid_list):
    names = ibs.get_annot_name_texts(aid_list)
    from collections import defaultdict

    name_aids = defaultdict(list)
    for aid, name in zip(aid_list, names):
        name_aids[name].append(aid)
    return name_aids


@register_ibs_method
def pie_new_accuracy(ibs, aid_list, min_sights=3, max_sights=10):
    aids = subset_with_resights_range(ibs, aid_list, min_sights, max_sights)
    ranks = ibs.pie_mass_accuracy(aids)
    accuracy = ibs.accuracy_at_k(ranks)
    logger.info(
        'Accuracy at k for annotations with %s-%s sightings:' % (min_sights, max_sights)
    )
    logger.info(accuracy)
    return accuracy


def _invert_dict(d):
    from collections import defaultdict

    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return dict(inverted)


# Careful, this returns a different ibs than you sent in
def pie_testdb_ibs():
    testdb_name = 'manta-test'
    try:
        ans_ibs = wbia.opendb(testdb_name)
        aids = ans_ibs.get_valid_annots()
        assert len(aids) > 3
        return ans_ibs
    except Exception:
        logger.info("PIE testdb does not exist; creating it with PIE's example images")

    ans_ibs = wbia.opendb(testdb_name, allow_newdir=True)

    test_image_folder = os.path.join(_PLUGIN_FOLDER, 'examples/manta-demo/test')
    test_images = os.listdir(test_image_folder)
    test_images = [fname for fname in test_images if fname.lower().endswith('.png')]
    test_images = sorted(test_images)

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
