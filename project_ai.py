import os
import numpy as np
import random
import sc2.ids
import sc2
from sc2.constants import *
from sc2 import run_game, Race, maps, Difficulty, Result
from sc2.player import Bot, Computer, Human
from sc2.ids import ability_id
import cv2
import math
import pickle
import time
import keras

HEADLESS = False



class Terran(sc2.BotAI):
    def __init__(self, use_model = True):
        self.train_data = []
        self.do_something = 0
        self.use_model = use_model
        if self.use_model:
            self.model = keras.models.load_model("BasicCNN-10-epochs-0.0001-LR-STAGE1")

    def save_game_data(self, number):
        save_directory = r'C:\Users\begzh\OneDrive\Desktop\sc2_ai\train_data'
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, f'game{number}.npy')
        np.save(file_path, np.array(self.train_data)) # bullshit

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_worker()
        await self.build_geyser()
        await self.expand()
        await self.first_marine()
        await self.build_techlab()
        await self.build_ffactory()
        await self.upgrade_shields()
        await self.build_bay()
        await self.bay_attack1()
        await self.build_freactor()
        await self.build_startport()
        await self.train_marines()
        await self.barrackfactory_replace()
        await self.build_supply()
        await self.land_buildings()
        await self.train_tanks()
        await self.build_starport_reactor()
        await self.train_medivacs()
        await self.train_bio()
        await self.medivac_move()
        await self.build_workers()
        await self.tank_mod()
        await self.set_rally()
        await self.build_barracks()
        await self.build_techlabs()
        await self.train_marouders()
        await self.train_after3_base()
        await self.upgrade_armor()
        await self.armory()
        await self.upgrade_second()
        await self.decision()
        await self.new_units_management()
        await self.timing_attack()
        await self.intel()

        self.iteration = iteration

        for depo in self.units(SUPPLYDEPOT).ready:
            for unit in self.known_enemy_units.not_structure:
                if unit.position.to2.distance_to(depo.position.to2) < 15:
                    break
            else:
                await self.do(depo(MORPH_SUPPLYDEPOT_LOWER))


        for depo in self.units(SUPPLYDEPOTLOWERED).ready:
            for unit in self.known_enemy_units.not_structure:
                if unit.position.to2.distance_to(depo.position.to2) < 10:
                    await self.do(depo(MORPH_SUPPLYDEPOT_RAISE))
                    break

        depot_placement_positions = self.main_base_ramp.corner_depots

        barracks_placement_position = None
        barracks_placement_position = self.main_base_ramp.barracks_correct_placement

        depots = self.units(SUPPLYDEPOT) | self.units(SUPPLYDEPOTLOWERED)

        if depots:
            depot_placement_positions = {d for d in depot_placement_positions if depots.closest_distance_to(d) > 1}

        if self.can_afford(SUPPLYDEPOT) and not self.already_pending(SUPPLYDEPOT):
            if len(depot_placement_positions) == 0:
                return
            target_depot_location = depot_placement_positions.pop()
            ws = self.workers.gathering
            if ws:
                w = ws.random
                await self.do(w.build(SUPPLYDEPOT, target_depot_location))

        if depots.ready.exists and self.can_afford(BARRACKS) and not self.already_pending(BARRACKS):
            if self.units(STARPORT).amount == 0:
                if self.units(BARRACKS).amount + self.already_pending(BARRACKS) > 0:
                    return
                ws = self.workers.gathering
                if ws and barracks_placement_position: 
                    w = ws.random
                    await self.do(w.build(BARRACKS, barracks_placement_position))

        if iteration == 5:
            scout_scv = self.units(SCV).first
            scout_target = self.enemy_start_locations[0]
            await self.do(scout_scv.move(scout_target))




    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, 
 CYBERNETICSCORE, STARGATE, VOIDRAY'''
        draw_dict = {
                     NEXUS: [15, (0, 255, 0)],
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],

                     VOIDRAY: [3, (255, 100, 0)],
                     #OBSERVER: [3, (255, 255, 255)],
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "supplydepot", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)


    

    async def build_worker(self):
        for commandcenter in self.units(COMMANDCENTER).ready.noqueue:
            if self.can_afford(SCV):
                if self.workers.amount < 21:
                    await self.do(commandcenter.train(SCV))

    async def build_workers(self):
        for cc in self.units(COMMANDCENTER).ready.noqueue:
            if self.units(COMMANDCENTER).amount == 2:
                if self.workers.amount < 42:
                    await self.do(cc.train(SCV)) 
   
    async def build_supply(self):
        if self.supply_left < 4 and not self.already_pending(SUPPLYDEPOT):
            cc = self.units(COMMANDCENTER).ready
            if cc.exists:
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near = cc.first)

    async def build_geyser(self):
        if self.units(BARRACKS).exists or self.already_pending(BARRACKS):
            for cc in self.units(COMMANDCENTER).ready:
                vgs = self.state.vespene_geyser.closer_than(15.0, cc)
                for vg in vgs:
                    if not self.can_afford(REFINERY):
                        break
                    w = self.select_build_worker(vg.position)
                    if not self.units(REFINERY).closer_than(1.0, vg).exists:
                        await self.do(w.build(REFINERY, vg))

    async def expand(self):
        if self.can_afford(COMMANDCENTER) and self.units(COMMANDCENTER).amount < 2:
            if self.supply_army > 0:
                await self.expand_now()

    async def first_marine(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE):
                if self.supply_army < 1: 
                    await self.do(barrack.train(MARINE))

    async def build_techlab(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(BARRACKSTECHLAB) and self.units(BARRACKS).amount == 1:
                if self.supply_army > 0:
                    if barrack.add_on_tag == 0:
                        await self.do(barrack.build(BARRACKSTECHLAB))
    
    async def build_ffactory(self):
        if self.can_afford(FACTORY) and self.units(COMMANDCENTER).amount == 2:
            if self.units(FACTORY).amount < 1 and self.units(STARPORT).amount == 0:
                if not self.already_pending(FACTORY):
                        cc = self.units(COMMANDCENTER).first
                        barrack = self.units(BARRACKS).first
                        if cc.position.y > self.game_info.map_center.y:
                            position = barrack.position.offset((0, 3))
                        elif cc.position.y < self.game_info.map_center.y:
                            position = barrack.position.offset((0, -3))
                        await self.build(FACTORY, position)

    async def upgrade_shields(self):
        for barracks in self.units(BARRACKS).ready:
            if barracks.add_on_tag:
                add_on = self.units.find_by_tag(barracks.add_on_tag)
                if add_on.type_id == BARRACKSTECHLAB:
                    await self.do(add_on(RESEARCH_COMBATSHIELD))

    async def build_bay(self):
        cc = self.start_location.position
        minerals = self.state.mineral_field.closest_to(cc).position
        if cc.position.y > self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((0, 3))
        elif cc.position.y > self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((0, 3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((0, -3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((0, -3))

        factory = self.units(FACTORY).ready
        if self.supply_used > 21 and self.units(ENGINEERINGBAY).amount < 1 and factory.exists:
            if self.can_afford(ENGINEERINGBAY) and not self.already_pending(ENGINEERINGBAY):
                await self.build(ENGINEERINGBAY, location)

    async def bay_attack1(self):
        upgrade_started = False
        for bay in self.units(ENGINEERINGBAY).ready:
            if self.can_afford(ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1) and not upgrade_started:
                await self.do(bay(ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1))
                upgrade_started = True

    async def build_freactor(self):
        for factory in self.units(FACTORY).ready.noqueue:
            if self.can_afford(FACTORYREACTOR):
                if factory and factory.add_on_tag == 0:
                    await self.do(factory.build(FACTORYREACTOR))

    async def build_startport(self):
        if self.can_afford(STARPORT) and self.units(FACTORY).exists:
            if not self.already_pending(STARPORT) and self.units(STARPORT).amount < 1:
                factory = self.units(FACTORY).first
                cc = self.units(COMMANDCENTER).first
                if cc.position.y > self.game_info.map_center.y:
                    position = factory.position.offset((0, 3))
                elif cc.position.y < self.game_info.map_center.y:
                    position = factory.position.offset((0, -3))
                await self.build(STARPORT, position)

    async def build_starport_reactor(self):
        for starport in self.units(STARPORT).ready.noqueue:
            if self.can_afford(STARPORTREACTOR):
                if starport and starport.add_on_tag == 0:
                    await self.do(starport.build(STARPORTREACTOR))

    async def train_marines(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if barrack.add_on_tag:
                add_on = self.units.find_by_tag(barrack.add_on_tag)
                if add_on.type_id == BARRACKSTECHLAB:
                    if self.units(MARINE).amount < 7:
                        await self.do(barrack.train(MARINE))

    async def barrackfactory_replace(self):
        barrack = self.units(BARRACKS).ready
        factory = self.units(FACTORY).ready
        if barrack.exists and factory.exists:
            barrack = barrack.first
            factory = factory.first
            if barrack.noqueue and factory.noqueue:
                if self.supply_army > 6 and self.units(BARRACKS).amount == 1:
                    add_on = self.units.find_by_tag(barrack.add_on_tag)
                    if add_on and add_on.type_id == BARRACKSTECHLAB:
                        await self.do(barrack(AbilityId.LIFT_BARRACKS))
                        await self.do(factory(AbilityId.LIFT_FACTORY))

    async def land_buildings(self):
        cc = self.units(COMMANDCENTER).first
        f_position = self.main_base_ramp.barracks_correct_placement
        if cc.position.y > self.game_info.map_center.y:
            b_position = f_position.position.offset((0, 3))
        elif cc.position.y < self.game_info.map_center.y:
            b_position = f_position.position.offset((0, -3))
        barrack = self.units(BARRACKSFLYING)
        factory = self.units(FACTORYFLYING)
        if barrack.exists and factory.exists:
            barrack = barrack.first
            factory = factory.first
            if barrack.is_flying and factory.is_flying:
                await self.do(barrack(AbilityId.LAND_BARRACKS, b_position))
                await self.do(factory(AbilityId.LAND_FACTORY, f_position))

    async def build_supply(self):
        if self.supply_left < 7 and not self.already_pending(SUPPLYDEPOT):
            if self.supply_used > 38:
                cc = self.units(COMMANDCENTER).first
                if self.supply_left < 4:
                    await self.build(SUPPLYDEPOT, near = cc)
                    await self.build(SUPPLYDEPOT, near = cc)
                else:
                    await self.build(SUPPLYDEPOT, near = cc)

    async def train_tanks(self):
        factory = self.units(FACTORY).noqueue
        if factory.exists:
            factory = factory.first
            if self.can_afford(SIEGETANK) and self.units(SIEGETANK).amount < 3:
                await self.do(factory.train(SIEGETANK))


    async def train_medivacs(self):
        starport = self.units(STARPORT)
        if starport.exists:
            starport = starport.first
            if self.can_afford(MEDIVAC):
                if self.units(MEDIVAC).amount < 2 and self.already_pending(MEDIVAC) < 1:
                    await self.do(starport.train(MEDIVAC))

    async def train_bio(self):
        barrack = self.units(BARRACKS)
        if barrack.exists:
            barrack = barrack.first
            add_on = self.units.find_by_tag(barrack.add_on_tag)
            if add_on and add_on.type_id == BARRACKSREACTOR:
                length = len(barrack.orders)
                if self.units(MARINE).amount < 24 and length < 2:
                    await self.do(barrack.train(MARINE))

    async def medivac_move(self):
        for medivac in self.units(MEDIVAC):
            target = self.enemy_start_locations[0]
            marine = self.units(MARINE).closest_to(medivac)
            await self.do(medivac.move(marine.position))
            for unit in self.known_enemy_units:
                if unit.position.to2.distance_to(medivac.position.to2) < 7:
                    await self.do(medivac.attack(target.position))
            for unit in self.units(MARINE):
                if unit.position.to2.distance_to(medivac.position.to2) < 7 and unit.health < 55:
                    await self.do(medivac.attack(target.position))

    async def tank_mod(self):
        for tank in self.units(SIEGETANK):
            for unit in self.known_enemy_units:
                tunk = self.units(SIEGETANK).closest_to(tank)
                if unit.position.to2.distance_to(tank.position.to2) < 10:
                    await self.do(tank(AbilityId.SIEGEMODE_SIEGEMODE))

        for tank in self.units(SIEGETANKSIEGED):
            enemies_nearby = self.known_enemy_units.closer_than(13, tank)
            if not enemies_nearby:
                await self.do(tank(AbilityId.UNSIEGE_UNSIEGE))

    async def set_rally(self):
        offensive_buildings = self.units(BARRACKS).ready | self.units(FACTORY).ready | self.units(STARPORT).ready
        for building in offensive_buildings:
            main_base = self.start_location
            expansions = sorted(self.expansion_locations, key=lambda x: x.distance_to(main_base))
            natural_expansion = expansions[1]
            await self.do(building(AbilityId.RALLY_BUILDING, natural_expansion.position.towards(self.game_info.map_center, 5.0)))

    async def build_barracks(self):
        cc = self.start_location.position
        if cc.position.y > self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.main_base_ramp.barracks_correct_placement.position.offset((5, 6))
            new_l = location.position.offset((0, 3))
        elif cc.position.y > self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.main_base_ramp.barracks_correct_placement.position.offset((-5, 6))
            new_l = location.position.offset((0, 3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.main_base_ramp.barracks_correct_placement.position.offset((5, -6))
            new_l = location.position.offset((0, -3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.main_base_ramp.barracks_correct_placement.position.offset((-5, -6))
            new_l = location.position.offset((0, -3))

        if self.units(STARPORT).ready.amount == 1 and self.units(BARRACKS).amount < 2:
            await self.build(BARRACKS, location)
        if self.units(STARPORT).ready.amount == 1 and self.units(BARRACKS).amount < 3:
            await self.build(BARRACKS, new_l)

    async def build_techlabs(self):
        for barrack in self.units(BARRACKS).ready:
            if not barrack.add_on_tag and self.units(BARRACKS).amount > 1:
                await self.do(barrack.build(BARRACKSTECHLAB))

    async def train_marouders(self):
        for barrack in self.units(BARRACKS).noqueue.ready:
            if self.units(BARRACKS).amount > 1:
                add_on = self.units.find_by_tag(barrack.add_on_tag)
                if add_on and add_on.type_id == BARRACKSTECHLAB:
                    if self.units(MARAUDER).amount < 8:
                        await self.do(barrack.train(MARAUDER))

    async def train_after3_base(self):
        if self.units(COMMANDCENTER).amount == 3:
            if self.supply_used < 150 :
                for barrack in self.units(BARRACKS).noqueue:
                    length = len(barrack.orders)
                    if self.supply_used < 140:
                        if length < 2:
                            await self.do(barrack.train(MARINE))

                for factory in self.units(FACTORY).noqueue:
                    if self.units(SIEGETANK).amount < 12:
                        await self.do(factory.train(SIEGETANK))

                for starport in self.units(STARPORT):
                    queue = len(starport.orders)
                    if queue < 2 and self.units(MEDIVAC).amount < 6:
                        await self.do(starport.train(MEDIVAC))

    async def upgrade_armor(self):
        for bay in self.units(ENGINEERINGBAY):
            if self.can_afford(ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2):
                await self.do(bay(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1))

    async def armory(self):
        cc = self.start_location.position
        minerals = self.state.mineral_field.closest_to(cc).position
        if cc.position.y > self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((3, 3))
        elif cc.position.y > self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((-3, 3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x > self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((3, -3))
        elif cc.position.y < self.game_info.map_center.y and cc.position.x < self.game_info.map_center.x:
            location = self.state.mineral_field.closest_to(cc).position.offset((-3, -3))

        if self.can_afford(ARMORY) and not self.already_pending(ARMORY):
            if self.units(ARMORY).amount < 1:
                await self.build(ARMORY, location)

    async def upgrade_second(self):
        for bay in self.units(ENGINEERINGBAY).noqueue:
            if self.units(ARMORY).exists:
                if self.can_afford(ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2):
                    await self.do(bay(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2))
                    await self.do(bay(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2))

    async def decision(self):
        if self.supply_used > 59:

            if self.model:
                prediction = self.model.predict([self.flipped.reshape([-1, 200, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 4)
                self.choice = choice

            if self.iteration > self.do_something:

                if choice == 0:
                    wait = 20
                    self.do_something = self.iteration + wait
                
                if choice == 1:
                    target = self.enemy_start_locations[0]
                    army = self.units(MARINE) | self.units(MARAUDER) | self.units(SIEGETANK)
                    for unit in army:
                        await self.do(unit.attack(target))
                    wait = 35
                    self.do_something = self.iteration + wait

                if choice == 2:
                    await self.expand_now()
                    wait = 1000
                    self.do_something = self.iteration + wait

                if choice == 3:
                    if self.units(LIBERATOR).amount < 2:
                        for starport in self.units(STARPORT).noqueue:
                            await self.do(starport.train(LIBERATOR))
                    wait = 65
                    self.do_something = self.iteration + wait

                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y, self.flipped])

                if choice == 3 and self.units(LIBERATOR).amount > 1:
                    target = self.enemy_start_locations[0]
                    army = self.units(MARINE) | self.units(MARAUDER) | self.units(SIEGETANK)
                    for unit in army:
                        await self.do(unit.attack(target))
                    for liberator in self.units(LIBERATOR).ready:
                        tank = self.units(SIEGETANK).closest_to(liberator)
                        await self.do(liberator.move(tank))
                        for enemy in self.known_enemy_units:
                            if enemy.position.to2.distance_to(liberator.position.to2) < 5:
                                await self.do(liberator(AbilityId.LIBERATORMORPHTOAG_LIBERATORAGMODE, enemy))

        

    async def new_units_management(self):
        if self.supply_used > 95:
            army = self.units(MARINE) | self.units(MARAUDER) | self.units(SIEGETANK)
            target = self.enemy_start_locations[0]
            struc_nearloc = self.known_enemy_structures.closer_than(20.0, target)
            for unit in army:
                if len(self.known_enemy_structures) > 0 and len(self.known_enemy_units) < 5:
                    struc = random.choice(self.known_enemy_structures)
                    await self.do(unit.attack(struc))
                elif not struc_nearloc:
                    await self.do(unit.attack(target.towards(self.game_info.map_center, 35.0)))
                else:
                    await self.do(unit.attack(target))

    async def timing_attack(self):
        if self.supply_used > 75:
            army = self.units(MARINE) | self.units(MARAUDER) | self.units(SIEGETANK)
            target = self.enemy_start_locations[0]
            third = target.towards(self.game_info.map_center, 35.0)
            struc_nearloc = self.known_enemy_structures.closer_than(17.0, target)
            struc_nearthird = self.known_enemy_structures.closer_than(17.0, third)
            cc = self.units(COMMANDCENTER).first
            if cc.position.y > self.game_info.map_center.y:
                second = math.ceil(self.enemy_start_locations[0].y + 30)
            elif cc.position.y < self.game_info.map_center.y:
                second = math.ceil(self.enemy_start_locations[0].y - 30)

            for unit in army:
                if len(self.known_enemy_structures) > 0 and len(self.known_enemy_units) < 5:
                    struc = random.choice(self.known_enemy_structures)
                    await self.do(unit.attack(struc))
                elif not struc_nearloc and struc_nearthird:
                    await self.do(unit.attack(third))
                elif not struc_nearthird and not struc_nearloc:
                    await self.do(unit.attack(second))
                else:
                    await self.do(unit.attack(target))
