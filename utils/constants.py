
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

VARS = {
    "edh": {
        "surf": [
            "u10", "v10", "t2m", "msl",
        ],
        "atmos": [
            "t", "u", "v", "q", "z",
        ]
    },
    "batch": {
        "atmos": {
            't': 'temperature',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'q': 'specific_humidity',
            'z': 'geopotential',
        },
        "surf": {
            '2t': '2m_temperature',
            '10u': '10m_u_component_of_wind',
            '10v': '10m_v_component_of_wind',
            'msl': 'mean_sea_level_pressure',
        },
    },
    "era5": {
        "surf": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "atmos": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ],
    },
}
