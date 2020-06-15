from __future__ import absolute_import, division, print_function
# import ibeis
from ibeis.control import controller_inject
import utool as ut
from .compute_db import hello

# imports the reid-manta stuff
# importlib.import_module('reid-manta/compute_db')


(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']


@register_ibs_method
def pie_hello(ibs):
    print('Hello again from PIE :)')
    hello()

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_deepsense._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
