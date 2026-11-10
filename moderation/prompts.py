"""Every prompt the models are given, in one place.

IMAGE_PROMPTS   SigLIP2: the text that defines each image category. Run `python model_download.py build-head`
                after any edit: the worker loads the built head, not this file.
CATEGORY_DETAILS, PROMPT_VERSION
                Qwen3Guard: detail text added to the model's own prompt. Bump PROMPT_VERSION after any edit
                (it is stored with every decision) and restart the worker.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# --- Images (SigLIP2) ---------------------------------------------------------
# Zero-shot prompts for the head. Rows in order: G-01..G-07, then safe:<slug>; each G-code row is
# the mean of its prompt embeddings. Each safe key is its OWN row (never reported; P(safe) = their
# sum), so a clean image only has to match one safe scene. Keys are lowercase letters, digits, _.
# Prompts are lowercased before encoding (SigLIP2 text towers are trained on lowercased text).
IMAGE_PROMPTS: Final[dict[str, list[str]]] = {
    # G-01 Violence and gore
    "G-01": [
        "a photo of a person with a bleeding wound",
        "a photo of a violent fight between people",
        "a photo of a person being beaten",
        "a graphic photo of a severe injury with blood",
        "a photo of a dead body at a crime scene",
    ],
    # G-02 Weapons
    "G-02": [
        "a photo of a handgun",
        "a photo of a rifle",
        "a photo of a person pointing a gun",
        "a photo of a knife held as a weapon",
        "a photo of a bomb or explosives",
    ],
    # G-03 Sexual content and nudity
    "G-03": [
        "a photo of a naked person",
        "an explicit sexual photo",
        "a photo of a person posing in underwear in a sexual way",
        "a nude photo",
    ],
    # G-04 Hate and harassment
    "G-04": [
        "a photo of a hate symbol",
        "a photo of a racist banner or slogan",
        "a photo of extremist propaganda",
        "a photo of a swastika",
        "a meme insulting a religious or ethnic group",
    ],
    # G-05 Self-harm
    "G-05": [
        "a photo of self-inflicted cuts on an arm",
        "a photo of self-harm scars on skin",
        "a photo of a noose",
        "a photo of a person attempting suicide",
    ],
    # G-06 Drugs and illegal goods
    "G-06": [
        "a photo of illegal drugs and white powder",
        "a photo of a person injecting drugs with a syringe",
        "a photo of marijuana joints",
        "a photo of pills and drug paraphernalia",
        "a photo of counterfeit money",
    ],
    # G-07 Crime and fraud
    "G-07": [
        "a screenshot of a phishing message asking for a password",
        "a photo of stolen credit cards",
        "a screenshot of a scam promising free money",
        "a photo of a burglary in progress",
        "a photo of a fake identity document",
    ],
    "safe:food": ["a photo of food on a plate"],
    "safe:mosque": ["a photo of a mosque"],
    "safe:calligraphy": ["a photo of arabic calligraphy"],
    "safe:family": ["a photo of a family together"],
    "safe:landscape": ["a photo of a landscape"],
    "safe:street": ["a photo of a city street"],
    "safe:selfie": ["a selfie of a smiling person"],
    "safe:pet": ["a photo of a pet"],
    "safe:shop": ["a photo of a shop front"],
    "safe:text_message": ["a screenshot of a text message"],
}

# --- Text (Qwen3Guard) --------------------------------------------------------
# The prompt is the model's own chat template plus these details: detail text next to two of its
# category names. The model is fine-tuned on that template. It ignores new category names, but it does
# use detail text next to the existing ones.

# Stored with every decision (worker.read_model_versions). Bump it whenever CATEGORY_DETAILS changes.
PROMPT_VERSION = "details-v3"
# Describe the trick, not the topic: real giveaways, shop fees and contact posts must stay clean.
CATEGORY_DETAILS: Mapping[str, str] = {
    "Non-violent Illegal Acts": (
        "selling or buying weapons, drugs or counterfeit goods; asking the reader to pay a fee, send money, "
        "or share an OTP, password, PIN or card number to receive a prize, job, refund or profit"
    ),
    "PII": "any person's ID number, home address, phone number or bank details shared in the post",
    "Unethical Acts": "asking people to harass, shame, insult or hunt down a person",
}
