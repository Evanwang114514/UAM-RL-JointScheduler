from at_obj.evtol.evtol_spec import eVTOLSpec

#补充亿航 等
#参数参考 .......
EVTOL_SPECS = {
    "JOBY_S4": eVTOLSpec(
        id = None,
        model="Joby S4",
        manufacturer="Joby Aviation",
        capacity = 4,
        max_speed = 200.0,
        range_km = 240.0,
        battery_capacity_kwh = 133.0,
        energy_consumption_kwh_per_km=0.55,
        min_reserve_ratio=0.20,
        charge_rate_kwh_per_min = 1,
    ),

    "LILIUM_JET": eVTOLSpec(
        id = None,
        model="Lilium Jet",
        manufacturer="Lilium",
        capacity = 6,
        max_speed = 280.0,
        range_km = 300.0,
        battery_capacity_kwh = 160.0,
        energy_consumption_kwh_per_km = 0.60,
        min_reserve_ratio=0.25,
        charge_rate_kwh_per_min = 1
    )
}