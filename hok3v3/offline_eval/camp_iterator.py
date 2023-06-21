import itertools
import random


def my_camp_iterator(heros_rl_camp, oppo_camp):

    camp_heros_rl = []
    camp_oppo_camp = []
    for i in range(len(heros_rl_camp)):
        camp_heros_rl = camp_heros_rl + ([[x["value"] for x in hero_list] for hero_list in itertools.product(*heros_rl_camp[i])])

    for i in range(len(oppo_camp)):
        camp_oppo_camp = camp_oppo_camp + ([[x["value"] for x in hero_list] for hero_list in itertools.product(*oppo_camp[i])])

    camps = [x for x in itertools.product(camp_heros_rl, camp_oppo_camp)]

    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def camp_iterator(dataset_name='level-1-1'):
    camp_list = []
    hero_list = [
        [{"name": "貂蝉", "value": "141"}, {"name": "诸葛亮", "value": "190"}],
        [{"name": "李元芳", "value": "173"}, {"name": "孙尚香", "value": "111"}],
        [{"name": "赵云", "value": "107"}, {"name": "钟无艳", "value": "117"}],
    ]
    for ii in range(len(hero_list[0])):
        for jj in range(len(hero_list[1])):
            for kk in range(len(hero_list[2])):
                camp_list.append([[hero_list[0][ii]], [hero_list[1][jj]], [hero_list[2][kk]]])
    driver = 'norm'
    if 'multi_hero_oppo' in dataset_name:
        driver = 'multi_hero_oppo'
    elif 'multi_hero' in dataset_name:
        driver = 'multi_hero'
    elif 'multi_oppo' in dataset_name:
        driver = 'multi_oppo'
    if driver == "norm":
        ally_camp = [camp_list[0]]
        oppo_camp = [camp_list[0]]
    elif driver == "multi_hero_oppo":
        ally_camp = camp_list
        oppo_camp = camp_list
    elif driver == "multi_hero":
        ally_camp = camp_list
        oppo_camp = [camp_list[0]]
    elif driver == "multi_oppo":
        ally_camp = [camp_list[0]]
        oppo_camp = camp_list
    else:
        raise Exception("Unknown camp driver: %s" % driver)

    return ally_camp, oppo_camp
