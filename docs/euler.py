import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tinkering with Eulerian Tours
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy
    import toytree
    import toyplot
    import seaborn
    import logging
    import time

    # Enable all messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    from itertools import combinations

    from SuchTree import SuchTree
    from quarimo import Tree, Forest

    treefile  = 'Projects/quarimo/data/gopher.tree'
    ensembleA = 'Projects/quarimo/data/iqtree_capsid.ufboot'
    ensembleB = 'Projects/quarimo/data/iqtree_dnapol.ufboot'
    ensembleC = 'Projects/quarimo/data/iqtree_rnapol1.ufboot'
    return (
        Forest,
        SuchTree,
        Tree,
        combinations,
        ensembleA,
        ensembleB,
        ensembleC,
        mo,
        numpy,
        time,
        toytree,
        treefile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One tree, two tree, `PhyloTree`, `SuchTree`

    For one tree toplogoy, check that `PhyloTree` and `SuchTree`

    - Have the same namespace
    - Agree on quartet topologies
    - Correctly handle multifrucations
    - Agree on pairwise distances
    - Don't crash
    """)
    return


@app.cell
def _(toytree, treefile):
    t = toytree.tree( treefile )
    canvas, axis, mark = t.draw( tip_labels_align=True, layout='unrooted' )
    #toytree.save( canvas, '/tmp/gophers.pdf' )
    canvas
    return


@app.cell
def _(SuchTree, Tree, combinations, numpy, treefile):
    ST = SuchTree( treefile )
    QT = Tree( open(treefile).read() )

    assert QT.quartet_topology( 'Ccas',
                                'Ocav',
                                'Ohet',
                                'Oche' ) == QT.quartet_topology( 'Ccas',
                                                                 'Ocav',
                                                                 'Oche',
                                                                 'Ohet' )
    topos = [ ( ST.quartet_topology( a,b,c,d ),
                QT.quartet_topology( a,b,c,d ) )
              for a,b,c,d in combinations( ST.leaf_names, 4 ) ]

    for s,p in topos :
        assert s==p

    names = ST.leaf_names
    dst = numpy.array( ST.distances_by_name( list( combinations( names, 2 ) ) ) )

    dpt = numpy.array( [ QT.branch_distance( a, b )
            for a,b
            in combinations( names, 2 ) ] )

    print( f'Largest distance error : {abs( dst - dpt ).max()}' )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All together now

    Test tree ensembles.
    """)
    return


@app.cell
def _(Forest, SuchTree, combinations, ensembleA, numpy):
    print( 'Loading quarimo forest...' )
    QF = Forest( [ t for t in open( ensembleA ) ] )

    print( 'Computing pairs...')
    pairs = list( combinations( QF.global_names, 2) )

    print( 'Loading SuchTree forest...' )
    SF = [ SuchTree(t) for t in open( ensembleA ) ]

    print( 'Calculating branch distances with quarimo forest...' )
    Dp = numpy.array( [ QF.branch_distance( a, b ) for a,b in pairs ] )

    print( 'Calculating branch distances with SuchTree forest...' )
    Ds = numpy.array( [ T.distances_by_name( pairs ) for T in SF ] ).T

    distance_error = Dp - Ds 

    print( f'Quarimo   :\n   min : {Dp.min()}\n   max : {Dp.max()}' )
    print( f'SuchTree  :\n   min : {Ds.min()}\n   max : {Ds.max()}' )
    print( f'max error : {abs( distance_error ).max()}' )
    return


@app.cell
def _(Forest, combinations, ensembleA, ensembleB, ensembleC, time):
    QFL = Forest( { 'A' : [ t for t in open( ensembleA ) ],
                    'B' : [ t for t in open( ensembleB ) ],
                    'C' : [ t for t in open( ensembleC ) ] } )

    quartets = list( combinations( QFL.global_names, 4 ) )[:10000]

    print( f'Testing topologies of {len(quartets)} quartets in an ensemble of {QFL.n_trees}...' )

    t0 = time.time()
    qd = QFL.quartet_topology( quartets, steiner=True, backend='best' )
    t1 = time.time()

    elapsed = t1 - t0
    problem_size = len(quartets)*QFL.n_trees

    print( f'Completed {problem_size} in {t1-t0:.2f} seconds at {problem_size/elapsed:.2f} quartets per second' )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
