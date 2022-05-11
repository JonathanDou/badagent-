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
from tier1_agent import LearningAgent, SimpleAgent, BetterAgent, AdaptiveAgent

from reinforce import Reinforce, Reinforce2 

import numpy as np
import tensorflow as tf 

import math 

warnings.simplefilter("ignore")

#USE THIS AGENT!!!! 
#agent uses two models 

#first model for proposing contracts second one for responding
#both model takes in the current offer as a state

#model = Reinforce(2000,1000)
#model2 = Reinforce2(2000,2)


class GIGASIGMASUPERAGENT1337(OneShotAgent):
    """
    My Agent

    Implement the methods for your agent in here!
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = Reinforce(2000,1000)
        self.model2 = Reinforce2(2000,2)
        
        self.model.load_weights("weights")
        self.model2.load_weights("weights2")

    def before_step(self):
        
       self.actions = []
       self.actions2 = []
       self.mystates = []
       self.mystates2 = []
       self.rewards = []
       self.rewards2 = []
       
       self.contracts = []
       
       self.currentoffers = {}
       
       self.ufun.find_limit(True)
       self.ufun.find_limit(False)
      

    def on_negotiation_success(self, contract, mechanism):     
        self.contracts.append(contract)
        
    def generate_offer(self, state):
        
        #generates an offer using the first model
        self.mystates.append(state)
            
        probs = self.model.call(tf.expand_dims(state, 0))
        indices = np.arange(len(probs[0]))
        
        probs = probs.numpy()[0]
        probs /= probs.sum()
        
        action = np.random.choice(indices, p=probs)
        
        quantity = action // 100
        price = action % 100
        
        self.actions.append(action)
            
        return quantity, price
    
    def generate_response(self, state):
        
        #generates a response using the second model
        
        self.mystates2.append(state)
            
        probs = self.model2.call(tf.expand_dims(state, 0))
        indices = np.arange(len(probs[0]))
        
        probs = probs.numpy()[0]
        probs /= probs.sum()
        
        action = np.random.choice(indices, p=probs)
        
        self.actions2.append(action)
            
        return action
     
    #calls 
    def propose(self, negotiator_id: str, state) -> "Outcome":
        
       #preposes a contract using the first model
       
       state2 = np.zeros(2000)
       buy = True 
       
       ami = self.get_nmi(negotiator_id)
       
       if self._is_selling(ami):
           buy = False 
           
       if negotiator_id in self.currentoffers:
           
           state2 = self.currentoffers[negotiator_id]
       
       else:
           
           #make new contract, start contract high if selling, low if buying
           
           if buy:
               
               state2[1900] = 1
           else:
               
               state2[999] = 1 
     
       quantity, price = self.generate_offer(state2)
       
       self.rewards.append(0)
       
       return (quantity, self.awi.current_step, price)

    def respond(self, negotiator_id, state, offer):
        
       #make a response using the second model, response is 0 or 1
        
       buy = 1
       
       ami = self.get_nmi(negotiator_id)
       
       if self._is_selling(ami):
           buy = 0
        
       noffer = np.zeros(2000)
       noffer[offer[0]*100 + offer[2] + 1000*buy] = 1
       
       self.currentoffers[negotiator_id] = noffer
       
       response = self.generate_response(noffer)
       
       self.rewards2.append(0)
       
       if response == 1:
           return ResponseType.ACCEPT_OFFER
       else:
           return ResponseType.REJECT_OFFER
       
    def _is_selling(self, ami):
       return ami.annotation["product"] == self.awi.my_output_product

    def step(self):
        
        #reward is actual utility - avg utility, backpropogation using loss functions
        
        u = 0 
        avg = 0
        
        if len(self.contracts) > 0: 
            
            u = self.ufun.from_contracts(self.contracts)
            
            avg = (self.ufun.max_utility + self.ufun.min_utility)/2
        
        if len(self.mystates) > 0:
        
            self.rewards[len(self.rewards)-1] = u - avg
            
            with tf.GradientTape() as tape:
                
                loss = self.model.loss(self.mystates, self.actions, self.rewards)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        if len(self.mystates2) > 0:
            
            self.rewards2[len(self.rewards2)-1] = u - avg
            
            with tf.GradientTape() as tape2:
                
                loss2 = self.model2.loss(self.mystates2, self.actions2, self.rewards2)
            
            gradients2 = tape2.gradient(loss2, self.model2.trainable_variables)
            self.model2.optimizer.apply_gradients(zip(gradients2, self.model2.trainable_variables))
'''
#single model for badagent v1 agent
model3 = Reinforce(2000,1000)

#single model agent
class BadAgent(OneShotAgent):
    """
    My Agent

    Implement the methods for your agent in here!
    """
    

    def before_step(self):
       self.actions = []
       self.mystates = []
       self.rewards = []
       
       self.contracts = []
       
       self.currentoffers = {}
       
       self.ufun.find_limit(True)
       self.ufun.find_limit(False)

    def on_negotiation_success(self, contract, mechanism):     
        self.contracts.append(contract)
        
    def generate_offer(self, state):
        
        self.mystates.append(state)
            
        probs = model3.call(tf.expand_dims(state, 0))
        indices = np.arange(len(probs[0]))
        
        probs = probs.numpy()[0]
        probs /= probs.sum()
        
        action = np.random.choice(indices, p=probs)
        
        quantity = action // 100
        price = action % 100
        
        self.actions.append(action)
            
            
        return quantity, price
     

    def propose(self, negotiator_id: str, state) -> "Outcome":
        
       
       state2 = np.zeros(2000)
       buy = True 
       
       ami = self.get_nmi(negotiator_id)
       
       if self._is_selling(ami):
           buy = False 
           
       if negotiator_id in self.currentoffers:
           
           state2 = self.currentoffers[negotiator_id]
       
       else:
           
           if buy:
               
               state2[1000] = 1
           else:
               
               state2[0] = 1 
     
       quantity, price = self.generate_offer(state2)
       
       self.rewards.append(0)
       
       return (quantity, self.awi.current_step, price)

    def respond(self, negotiator_id, state, offer):
        
       buy = 1
       
       ami = self.get_nmi(negotiator_id)
       
       if self._is_selling(ami):
           buy = 0
        
       noffer = np.zeros(2000)
       noffer[offer[0]*100 + offer[2] + 1000*buy] = 1
       
       self.currentoffers[negotiator_id] = noffer
       
       response = self.generate_offer(noffer)
       
       self.rewards.append(0)
       
       if response[1] >= offer[2]:
           
           if buy: 
               return ResponseType.ACCEPT_OFFER
               
           return ResponseType.REJECT_OFFER

       else:
           
           if buy:
               return ResponseType.REJECT_OFFER
           return ResponseType.ACCEPT_OFFER
       
    def _is_selling(self, ami):
       return ami.annotation["product"] == self.awi.my_output_product

    def step(self):
        
        if len(self.contracts) > 0: 
            
            u = self.ufun.from_contracts(self.contracts)
            
            avg = (self.ufun.max_utility + self.ufun.min_utility)/2
            
            self.rewards[len(self.rewards)-1] = u - avg 
        
        if len(self.mystates) == 0:
            return
        
        with tf.GradientTape() as tape:
            
            loss2 = model3.loss(self.mystates, self.actions, self.rewards)

        
        gradients = tape.gradient(loss2, model3.trainable_variables)
        model3.optimizer.apply_gradients(zip(gradients, model3.trainable_variables))
'''
def main():
    """
    For more information:
    http://www.yasserm.com/scml/scml2020docs/tutorials/02.develop_agent_scml2020_oneshot.html
    """

    # TODO: Add/Remove agents from the list below to test your agent against other agents!
    #       (Make sure to change MyAgent to your agent class name)
    agents = [GIGASIGMASUPERAGENT1337, LearningAgent]
    agent1 = "GIGASIGMASUPERAGENT1337"
    agent2 = "LearningAgent"
    
    p1 = 0
    p2 = 0
    
    score1 = []
    score2 = []
    
    for i in range(150):
        
        world, ascores, tscores = try_agents(agents, draw=False) # change draw=True to see plot
    
        # TODO: Uncomment/Comment below to print/hide the individual agents' scores
        # print_agent_scores(ascores)
    
        # TODO: Uncomment/Comment below to print/hide the average score of for each agent type
        print("Scores: ")
        print_type_scores(tscores)
        
        if not agent2 in tscores:
            continue
        
        if tscores[agent1] > tscores[agent2]:
            p1 += 1
        else:
            p2 += 1
        
        print("Score " + agent1 + "/" + agent2 + ": " + str(p1) + "/" + str(p2))
        
        score1.append(tscores[agent1])
        score2.append(tscores[agent2])

        # TODO: Uncomment/Comment below to print/hide the exogenous contracts that drive the market
        #print(analyze_contracts(world))
    
    
    a = np.arange(150)

    plt.plot(a, score1, label = agent1)
    plt.plot(a, score2, label = agent2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()