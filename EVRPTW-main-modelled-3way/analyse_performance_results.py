import pstats
p = pstats.Stats('./results/main.prof')
p.sort_stats('time').print_stats(15)
# See for further analysis https://gertingold.github.io/tools4scicomp/profiling.html
