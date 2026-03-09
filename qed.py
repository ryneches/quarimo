import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from SuchTree import SuchTree
    from quarimo import Tree, Forest, Quartets

    from math import factorial, log, isclose
    from scipy.stats import pearsonr, kendalltau
    from itertools import combinations, permutations, chain
    from collections import defaultdict, Counter

    import seaborn
    import os
    import random
    import numpy

    datasets = [ { 'gene_a' : 'GVOG04832', 'gene_b' : 'GVOG06851' },
                 { 'gene_a' : 'GVOG06814', 'gene_b' : 'GVOG00888' },
                 { 'gene_a' : 'GVOG03189', 'gene_b' : 'GVOG06851' },
                 { 'gene_a' : 'GVOGm1795', 'gene_b' : 'GVOG03675' },
                 { 'gene_a' : 'GVOGm0626', 'gene_b' : 'GVOG10625' },
                 { 'gene_a' : 'GVOG03218', 'gene_b' : 'GVOG05492' },
                 { 'gene_a' : 'GVOGm0003', 'gene_b' : 'GVOG06851' },  # interesting case with conflicts
                 { 'gene_a' : 'GVOGm0003', 'gene_b' : 'GVOG03858' } ]

    prefix = '/home/russell/Projects/kizuchi-ncldvs/77-849/taxon_trees/'
    return (
        Counter,
        Forest,
        Quartets,
        SuchTree,
        combinations,
        datasets,
        numpy,
        os,
        pearsonr,
        permutations,
        prefix,
        seaborn,
    )


@app.cell
def _(SuchTree, combinations, datasets, os, pearsonr, prefix, seaborn):
    gene_a = datasets[5]['gene_a']
    gene_b = datasets[5]['gene_b']

    F1 = [ SuchTree( nwk ) for nwk in open( os.path.join( prefix, '{gene}.ufboot'.format( gene=gene_a ) ) ) ]
    F2 = [ SuchTree( nwk ) for nwk in open( os.path.join( prefix, '{gene}.ufboot'.format( gene=gene_b ) ) ) ]

    T1 = SuchTree( os.path.join( prefix, '{gene}.ufboot'.format( gene=gene_a ) ) )
    T2 = SuchTree( os.path.join( prefix, '{gene}.ufboot'.format( gene=gene_b ) ) )

    links = [ (a,b) for a,b in combinations( set( T1.leafs.keys() ) & set( T2.leafs.keys() ), 2 ) ]

    n = len( set.intersection( *[ set( T0.leafs.keys() ) for T0 in F1 ],
                               *[ set( T0.leafs.keys() ) for T0 in F2 ] ) )

    X = T1.distances_by_name( links )
    Y = T2.distances_by_name( links )

    r,p=pearsonr( X, Y )

    print( 'Pearson\'s r : {r} (p={p})'.format( r=str(r), p=str(p) ) )
    print( 'tree links  :', n )

    seaborn.jointplot( x=X, y=Y, kind='reg', height=4 )
    return F1, F2, gene_a, gene_b


@app.cell
def _(Counter, numpy, permutations):
    def quartet_topologies( quartet ) :
        'compute the three possible topologies of four taxa'
        # not the fastest way to do this...
        return frozenset( frozenset((frozenset((a,b)),frozenset((c,d))))
                          for a,b,c,d in permutations( quartet, 4 ) )

    def unpack_vectors( v ) :
        return numpy.array(v).T

    def quartet_frequencies( F1, F2, quartets ) :
        '''
        From two ensembles of trees, count the occurances of each possible topology of the
        given quartets. Returns a dictionary of three tuples of two integers :
    
            {a,b,c,d} : ( (c0a,c0b), (c1a,c1b), (c2a,c2b) )

        where c0a, c1a and c2a are the counts of the three topologies in the first ensemble, and
        c0a, c1a and c2a are the counts of the corresponding topologies in second ensemble.
        '''
        
        Q1 = [ Counter(q) for q in zip( *[ T.quartet_topologies_by_name( quartets ) for T in F1 ] ) ]
        Q2 = [ Counter(q) for q in zip( *[ T.quartet_topologies_by_name( quartets ) for T in F2 ] ) ]

        data = {}
        for q,c1,c2 in zip( quartets, Q1, Q2 ) :
            topologies = quartet_topologies( q )
            for t in topologies :
                if not t in c1 : c1[t] = 0
                if not t in c2 : c2[t] = 0
    
            data[ q ] = sorted( [ (c1[key], c2[key] ) for key in topologies ],
                                reverse=True,
                                key=lambda x : max( [ x[0], x[1] ] ) )
            
        return [ unpack_vectors(v) for v in data.values() ]

    return (quartet_frequencies,)


@app.cell
def _(F1, F2, combinations, numpy, quartet_frequencies):
    N = 10

    shared_leafs = set.intersection( *[ set( T0.leafs.keys() ) for T0 in F1 ],
                                     *[ set( T0.leafs.keys() ) for T0 in F2 ] )

    quartets = list( combinations( shared_leafs, 4 ) )[:N]

    print( numpy.array( quartet_frequencies( F1, F2, quartets ) ) )
    return N, quartets


@app.cell
def _(Forest, N, Quartets, gene_a, gene_b, os, prefix, quartets):
    F = Forest( { gene_a : [ t for t in open( os.path.join( prefix, f'{gene_a}.ufboot' ) ) ],
                  gene_b : [ t for t in open( os.path.join( prefix, f'{gene_b}.ufboot' ) ) ] } )

    Q = Quartets( F, seed=quartets, count=N )

    print( F.quartet_topology( Q ) )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
