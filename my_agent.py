"""
CS1440/CS2440 Negotiation Final Project

reference: http://www.yasserm.com/scml/scml2020docs/tutorials.html
author: ninagawa

File to implement negotiation agent

"""

import warnings

from typing import Any
from typing import Dict
from typing import List

from negmas import Contract
from negmas import MechanismState
from negmas import NegotiatorMechanismInterface
from negmas import Outcome
from negmas import ResponseType

from print_helpers import *
from tier1_agent import LearningAgent

warnings.simplefilter("ignore")


# TODO: Change the class name to something unique. This will be the name of your agent.
class MyAgent(OneShotAgent):
    """
    My Agent

    Implement the methods for your agent in here!
    """

    def init(self):
        """
        Called once after the AWI (Agent World Interface) is set.

        Remarks:
            - Use this for any proactive initialization code.
        """
        # TODO

    def before_step(self):
        """
        Called once every day before running the negotiations

        """
        # TODO
        
        self.secured = 0

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        """
        Proposes an offer to one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step

        Returns:
            an outcome to offer.
        """
        # TODO
        
        return self.best_offer(negotiator_id)
    
    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value),
            quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        """
        Responds to an offer from one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step
            offer: The offer received.

        Returns:
            A response type which can either be reject, accept, or end negotiation.

        Remarks:
            default behavior is to accept only if the current offer is the same
            or has a higher utility compared with what the agent would have
            proposed in the given state and reject otherwise

        """
        # TODO
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )
    
    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product
    
    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """
        Called whenever a negotiation ends without agreement.

        Args:
            partners: List of the partner IDs consisting from self and the opponent.
            annotation: The annotation of the negotiation including the seller ID,
                        buyer ID, and the product.
            mechanism: The `NegotiatorMechanismInterface` instance containing all information
                       about the negotiation.
            state: The final state of the negotiation of the type `SAOState`
                   including the agreement if any.
        """
        # TODO

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """
        Called whenever a negotiation ends with agreement.

        Args:
            contract: The `Contract` agreed upon.
            mechanism: The `NegotiatorMechanismInterface` instance containing all information
                       about the negotiation that led to the `Contract` if any.
        """
        # TODO
        
        self.secured += contract.agreement["quantity"]

    def step(self):
        """
        Called every step.

        Remarks:
            - Use this for any proactive code  that needs to be done every
              simulation step.
        """
        # TODO


def main():
    """
    For more information:
    http://www.yasserm.com/scml/scml2020docs/tutorials/02.develop_agent_scml2020_oneshot.html
    """

    # TODO: Add/Remove agents from the list below to test your agent against other agents!
    #       (Make sure to change MyAgent to your agent class name)
    agents = [LearningAgent, MyAgent]
    world, ascores, tscores = try_agents(agents, draw=False) # change draw=True to see plot

    # TODO: Uncomment/Comment below to print/hide the individual agents' scores
    # print_agent_scores(ascores)

    # TODO: Uncomment/Comment below to print/hide the average score of for each agent type
    print("Scores: ")
    print_type_scores(tscores)

    # TODO: Uncomment/Comment below to print/hide the exogenous contracts that drive the market
    # print(analyze_contracts(world))


if __name__ == '__main__':
    main()