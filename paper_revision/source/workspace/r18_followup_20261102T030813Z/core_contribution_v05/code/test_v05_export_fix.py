import analyze_v05 as original
import analyze_v05_export_fix_v01 as fixed
p=dict(replan=0,timestamp_us=1550000,source_sha='fixture',choice=12,seconds=1.,base_seconds=.5,base_receipt=dict(map_supply=dict(trigger=True,attempts=[dict(admitted=True,index=12,eligible=True)])))
j=dict(index=1,route='fixture',seed=17,method='H-Map',slot=2)
try:original.mechanics([p],j,'confirmation')
except TypeError as e:assert 'slot'in str(e)
else:raise AssertionError('Regression fixture did not reproduce frozen export bug')
m,rows=fixed.mechanics([p],j,'confirmation');assert rows[0]['slot']==2 and rows[0]['supply_slot']==0 and rows[0]['selected_final']and m['attempts']==1 and m['admitted']==1
assert fixed.self_test()['passed']
print('PASS: duplicate field reproduced, physical execution slot and supply slot preserved separately; frozen official/statistical behavior unchanged')
