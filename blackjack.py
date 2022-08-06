from typing import Optional, Optional, Tuple
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from itertools import product


@dataclass
class Card:
    face: str
    value: int


class Deck:
    """
    An infinite deck of standard playing cards.
    """
    cards = [
        Card('A', 1),
        *[Card(str(i), i) for i in range(2, 11)],
        Card('J', 10),
        Card('Q', 10),
        Card('K', 10)
    ]

    def __init__(self, seed: Optional[int] = 0):
        if seed is not None:
            self.rand = random.Random(seed)
        else:
            self.rand = random

    def draw_card(self) -> Card:
        return self.rand.choice(self.cards)


class BlackjackAction(Enum):
    HIT = 0
    STICK = 1


@dataclass(frozen=True)
class BlackjackState:
    player_sum: int
    usable_ace: bool
    dealer_showing: int
    terminal: bool

    @classmethod
    @property
    def all_states(cls):
        """
        Return the set of all possible states.
        """
        # Player sums range from 2 to 30
        # Usable ace can be True or False
        # Dealer showing ranges from 1 to 10
        # State is either terminal or not
        return set(cls(*tup) for tup in product(range(2, 31), [False, True], range(1, 11), [False, True]))
            

@dataclass(frozen=True)
class BlackjackStateAction(BlackjackState):
    action: BlackjackAction

    @property
    def state(self) -> BlackjackState:
        """
        Return the state only.
        """
        return BlackjackState(self.player_sum, self.usable_ace, self.dealer_showing, self.terminal)


class Blackjack:
    """
    How to use:
    1. Intialize the class. `reset` is called implicitly.
    2. Call `play_action` until the game is over.
    3. If the game is over you must call `reset` to start a new one. You cannot call `play_action`
    Whether the game is over or not can be checked by the in_play attribute.
    """
    def __init__(self, deck_seed: Optional[int] = 0, verbose: bool = False):
        self.deck = Deck(deck_seed)
        self.verbose = verbose
        self.reset()


    def reset(self) -> Tuple[BlackjackState, int]:
        self.turn = 'player'  # keep track of whose turn it is
        self.player_cards = []
        self.dealer_cards = []
        self.deal_hands()
        self.in_play = True
        if self.player_sum == 21:  # Natural
            self.in_play = False
            return self.state, 1
        return self.state, 0

    @staticmethod
    def get_hand_sum(cards) -> int:
        """ Compute the value of a hand keeping in mind that Ace's can be 1 or 11.
        The logic tries to get the highest value hand while staying under 21.
        """
        non_ace_sum = sum(card.value for card in cards if card.face != 'A')
        # Add in the aces as 11s until
        num_aces = sum(card.face == 'A' for card in cards)
        hand_sum = non_ace_sum + num_aces
        for _ in range(num_aces):
            # If we can safely switch this ace's value to 11, do it.
            if hand_sum <= 11:
                hand_sum += 10
        return hand_sum

    @property
    def dealer_sum(self) -> int:
        return self.get_hand_sum(self.dealer_cards)

    @property
    def player_sum(self) -> int:
        return self.get_hand_sum(self.player_cards)

    @property
    def usable_ace(self) -> bool:
        if not any(card.face == 'A' for card in self.player_cards):
            return False
        elif sum(card.value for card in self.player_cards) >= 12:
            # There's at least one ace but it can't be switched to 11 (thereby adding 10) without going bust.
            return False
        else:
            return True

    @property
    def dealer_showing(self) -> Card:
        return self.dealer_cards[1]

    @property
    def state(self):
        """
        Return the state of the environment as viewed by an agent.
        """
        return BlackjackState(self.player_sum, self.usable_ace, self.dealer_showing.value, not self.in_play)

    def deal_hands(self):
        """
        Dealer and player are dealt two cards each. The second dealer card is showing.
        """
        self.player_cards = [self.deck.draw_card() for _ in range(2)]
        self.dealer_cards = [self.deck.draw_card() for _ in range(2)]
        if self.verbose:
            self.render()

    def play_action(self, action: BlackjackAction) -> Tuple[BlackjackState, int]:
        """
        Play an action from either stick or hit. Return the resulting state, and optionally a reward if the episode
        has terminated.
        """

        assert self.in_play, "Game is over. Call the reset method to start a new one."

        if action == BlackjackAction.STICK:
            # Dealer sticks on 17+ and hits otherwise.
            self.turn = 'dealer'
            while self.dealer_sum < 17:
                self.dealer_cards.append(self.deck.draw_card())
                if self.verbose:
                    self.render()
            self.in_play = False  # game will be over after this function call
            dealer_sum = self.dealer_sum
            player_sum = self.player_sum
            if dealer_sum > 21:
                return self.state, 1
            elif dealer_sum > player_sum:
                return self.state, -1
            elif dealer_sum < player_sum:
                return self.state, 1
            else:
                return self.state, 0
        elif action == BlackjackAction.HIT:
            self.player_cards.append(self.deck.draw_card())
            if self.verbose:
                self.render()
            if self.player_sum > 21:
                self.in_play = False  # game is over
                return self.state, -1
        return self.state, 0

    def render(self):
        print()
        if self.turn == 'dealer':
            print("Dealer:", [card.face for card in self.dealer_cards], "=", self.dealer_sum)
        else:
            print("Dealer:", ['x'] + [card.face for card in self.dealer_cards[1:]])
        print("Player:", [card.face for card in self.player_cards], "=", self.player_sum)


class BaseBlackjackAgent(ABC):
    """
    Base class for any agent that plays BlackJack.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def choose_action(self, state: BlackjackState) -> BlackjackAction:
        # Define a policy here
        pass


class BasicBlackjackAgent(BaseBlackjackAgent):
    """
    Agent sticks if their hand sums to some value above or equal to a given threshold, otherwise they hit.
    """

    def __init__(self, stick_threshold):
        self.stick_threshold = stick_threshold
        self.policy = {
            state: BlackjackAction.STICK if state.player_sum >= stick_threshold
            else BlackjackAction.HIT for state in BlackjackState.all_states}

    def choose_action(self, state: BlackjackState) -> BlackjackAction:
        return self.policy[state]


class RandomBlackjackAgent(BaseBlackjackAgent):
    def __init__(self, seed: int =0):
        rand = random.Random(seed)
        # policy is a mapping from a state to an action.
        # Initialize a random policy.
        self.policy = {state: rand.choice(list(BlackjackAction)) for state in BlackjackState.all_states}

    def choose_action(self, state: BlackjackState) -> BlackjackAction:
        return self.policy[state]
