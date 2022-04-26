"""
CS1440/CS2440 Negotiation Final Project

reference: http://www.yasserm.com/scml/scml2020docs/tutorials.html
author: ninagawa

File to run negotiation tournament

"""

import warnings

from scml.oneshot import *
from scml.scml2020.utils import anac2022_oneshot
from tier1_agent import LearningAgent
from my_agent import MyAgent # TODO: change the import agent name to your agent class name

warnings.simplefilter("ignore")


def shorten_names(results):
    """
    method to make agent types more readable
    """
    results.score_stats.agent_type = results.score_stats.agent_type.str.split(".").str[-1].str.split(":").str[-1]
    results.kstest.a = results.kstest.a.str.split(".").str[-1].str.split(":").str[-1]
    results.kstest.b = results.kstest.b.str.split(".").str[-1].str.split(":").str[-1]
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(".").str[-1].str.split(":").str[-1]
    results.scores.agent_type = results.scores.agent_type.str.split(".").str[-1].str.split(":").str[-1]
    results.winners = [_.split(".")[-1].split(":")[-1] for _ in results.winners]
    return results


def main():
    # TODO: Modify this list to include/delete your agents! (Make sure to change MyAgent to your agent class name)
    tournament_types = [RandomOneShotAgent, LearningAgent, GreedyOneShotAgent, MyAgent]

    # TODO: Modify the parameters to see how your agent performs in different settings
    results = anac2022_oneshot(
        competitors=tournament_types,
        n_configs=10, # number of different configurations to generate
        n_runs_per_world=1, # number of times to repeat every simulation (with agent assignment)
        n_steps = 10, # number of days (simulation steps) per simulation
        print_exceptions=True,
    )

    results = shorten_names(results)

    # TODO: Uncomment/Comment below to print/hide the winner of the tournament
    print("Winners: ", results.winners, "\n")

    # TODO: Uncomment/Comment below to print/hide stats of the tournament
    print(results.score_stats)


if __name__ == '__main__':
    """
    To see more things you can do when running a tournament (ie. plots) go to:
    http://www.yasserm.com/scml/scml2020docs/tutorials/01.run_scml2020.html#running-a-one-shot-tournament
    
    """
    main()