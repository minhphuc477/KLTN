from bench.grid_solvers import astar, jps

def run_smoke():
    g = [[0,0,0],[0,1,0],[0,0,0]]
    p_a, n_a = astar(g, (0,0), (2,2))
    p_j, n_j = jps(g, (0,0), (2,2))
    print('astar', len(p_a), n_a)
    print('jps', len(p_j), n_j)

if __name__ == '__main__':
    run_smoke()
