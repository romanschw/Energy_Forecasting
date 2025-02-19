import numpy as np
import pandas as pd

def compute_cost(fuels: dict, plants: dict)->float:
    if plants["type"] == "gasfired":
        cost = fuels["gas(euro/MWh)"] / plants["efficiency"]
    elif plants["type"] == "turbojet":
        cost = fuels["kerosine(euro/MWh)"] / plants["efficiency"]
    else:
        cost = fuels["wind(%)"] #Wind are considered generating power at zero price.
    return cost

def is_tractable(remaining_allocation: float, remaining_plants: dict, fuels) -> bool:

    temp_remaining = remaining_allocation

    for plant in remaining_plants:

        if plant["type"]=="windturbine":

            alloc = min(plant["pmax"] * fuels["wind(%)"]/100, remaining_allocation)

            if alloc == remaining_allocation:

                break
        else:
            if plant["pmin"] > remaining_allocation:
                continue

            alloc = min(plant["pmax"], remaining_allocation)
        
        temp_remaining -= round(alloc, 1)

    return temp_remaining <= 0

            

def greedy_compute(payload: dict) -> dict:

    response = []

    sorted_plants = sorted(payload["powerplants"], key=lambda x: (x["type"]=="windturbine", compute_cost(payload["fuels"], x)))
    
    remaining_load = payload["load"]

    for i, plant in enumerate(sorted_plants):

        plant_type = plant["type"]

        if (plant_type=="windturbine"):

            alloc = round(min(plant["pmax"] * payload["fuels"]["wind(%)"] / 100, remaining_load), 1)


        else:

            if plant["pmin"] > remaining_load:

                continue

            max_alloc = min(plant["pmax"], remaining_load)
            min_alloc = (plant["pmin"] if plant["pmin"] < remaining_load else 0)

            for x in [round(v, 1) for v in np.arange(max_alloc, min_alloc, -0.1)]:

                if is_tractable(round(remaining_load - x, 1), sorted_plants[i+1:], payload["fuels"]) or round(remaining_load - x, 1) <= 0:

                    alloc = x
        remaining_load -= round(alloc, 1)
        response.append({"name": plant["name"], "p": alloc})

    return response
                


# def calculate_production_plan(payload: dict):

if __name__ == "__main__":
    payload = {
  "load": 910,
  "fuels":
  {
    "gas(euro/MWh)": 13.4,
    "kerosine(euro/MWh)": 50.8,
    "co2(euro/ton)": 20,
    "wind(%)": 60
  },
  "powerplants": [
    {
      "name": "gasfiredbig1",
      "type": "gasfired",
      "efficiency": 0.53,
      "pmin": 100,
      "pmax": 460
    },
    {
      "name": "gasfiredbig2",
      "type": "gasfired",
      "efficiency": 0.53,
      "pmin": 100,
      "pmax": 460
    },
    {
      "name": "gasfiredsomewhatsmaller",
      "type": "gasfired",
      "efficiency": 0.37,
      "pmin": 40,
      "pmax": 210
    },
    {
      "name": "tj1",
      "type": "turbojet",
      "efficiency": 0.3,
      "pmin": 0,
      "pmax": 16
    },
    {
      "name": "windpark1",
      "type": "windturbine",
      "efficiency": 1,
      "pmin": 0,
      "pmax": 150
    },
    {
      "name": "windpark2",
      "type": "windturbine",
      "efficiency": 1,
      "pmin": 0,
      "pmax": 36
    }
  ]
}
    
response = greedy_compute(payload=payload)
print(response)