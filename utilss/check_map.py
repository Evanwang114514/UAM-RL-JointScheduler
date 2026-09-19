import sys
from pathlib import Path
parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))

from at_obj.map.map import Map
sim_map = Map()

from at_obj.vertiport.vertiport_builder import UamBuilder

uam = UamBuilder()
for i in uam.vertiport_list:
    print(uam.vertiport_list[i].vertiport_position)

from at_obj.scenario import Scenario

Scenario = Scenario()
uam_info = Scenario.state['UAM'][0].get_state()
print(uam_info)