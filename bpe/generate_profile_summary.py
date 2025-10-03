import pstats
from pstats import SortKey

# Load the profile
profile_path = 'logs/profile_tinystoriesv2_valid_baseline.prof'


p = pstats.Stats(profile_path)

print('='*80)
print('PROFILE SUMMARY: TinyStoriesV2 BPE Training')
print('='*80)
print()

# Get total time
total_time = p.total_tt
print(f'Total Runtime: {total_time:.3f} seconds ({total_time/60:.2f} minutes)')
print()

print('='*80)
print('TOP 20 FUNCTIONS BY CUMULATIVE TIME')
print('='*80)
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(20)

print()
print('='*80)
print('TOP 20 FUNCTIONS BY INTERNAL TIME (tottime)')
print('='*80)
p.sort_stats(SortKey.TIME)
p.print_stats(20)

print()
print('='*80)
print('TOP 20 FUNCTIONS BY CALL COUNT')
print('='*80)
p.sort_stats(SortKey.CALLS)
p.print_stats(20)

print()
print('='*80)
print('FUNCTIONS IN bpe.py ONLY')
print('='*80)
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats('bpe.py')