class eVTOLSpec:
    def __init__(self, id, model, manufacturer, capacity,
                 max_speed, range_km, battery_capacity_kwh,
                 min_reserve_ratio, energy_consumption_kwh_per_km,
                 charge_rate_kwh_per_min):
        self.id = id
        self.model = model
        #self.energy_per_km = energy_consumption_kwh_per_km
        self.manufacturer = manufacturer
        self.capacity = capacity
        self.max_speed = max_speed    
        self.range_km = range_km
        self.battery_capacity_kwh = battery_capacity_kwh
        self.min_reserve_ratio = min_reserve_ratio
        self.energy_consumption_kwh_per_km = energy_consumption_kwh_per_km
        self.charge_rate_kwh_per_min = charge_rate_kwh_per_min
