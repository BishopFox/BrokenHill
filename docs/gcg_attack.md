# How the greedy coordinate gradient (GCG) attack works

For someone who doesn't already have extensive experience working with large language models (LLMs), the GCG attack can seem like an arcane piece of alien technology that may never make sense. I know it did to me when I originally began work on this project. Reading [the "Universal and Transferable Adversarial Attacks on Aligned Language Models" paper](https://arxiv.org/abs/2307.15043) will likely feel like jumping into the deep end of the pool, and [reading the original authors' code](https://github.com/llm-attacks/llm-attacks) is unlikely to help because so much of the critical elements are either undocumented, or documented in a way that doesn't make sense without a strong LLM background. This document is my attempt to explain the attack in a way that's understandable to most people with some sort of information technology background. Any errors in the description are my own.

## Large language models: how do they work?

I'm not going to make you an LLM PhD here. For purposes of this discussion, all you need to know is that LLMs are very complex systems that examine the text they're given, then attempt to add more text at the end. They do this mostly based on approximate statistical likelihoods that they absorbed during their training. If they're configured in a way that allows the output to be different each time ("non-deterministic output"), there is also a random chance factor that can guide the course of the generated text down other paths. LLMs don't "reason", or "think", and they're certainly not self-aware, although their output can often seem to exhibit those qualities because they've absorbed so much information from sources written by humans.

If you're familiar with Markov chains, you can think of an LLM as being a very complex Markov chain generator, except that unlike a traditional Markov chain generator, the LLM is influenced by the entire set of text it's received so far, not just the most recent element.

For example, an LLM trained as a chatbot may receive the following text:

```
<|user|> Please tell me about Przybylski's Star.

<|assistant|>
```

Statistically, the most likely text to follow this is the assistant's response to the request "Please tell me about [Przybylski's Star](https://websites.umich.edu/~cowley/przyb.html)". More specifically, if the LLM has been trained or instructed to use friendly language, statistically the most likely next word will be something like "Sure", or an equivalent series of words like "I'd", then "be", "happy", and "to".

If you've seen the film *Equilibrium*, recall the explanation of the fictional "gun kata" fighting style. In that fictional world, statistical analysis of thousands of gunfights has been used to predict the most likely locations for opponents and their gunfire, so that a Grammaton cleric can pre-emptively shoot where an opponent is likely to be in the span of time it will take the bullet to reach that position. LLMs are sort of the text equivalent of that. In the film, Grammaton clerics also use the same statistical knowledge to position themselves and contort their body to avoid the paths of bullets being fired back at them. That aspect of the "gun kata" is not directly involved in typical LLM interaction or the GCG attack, but may be useful to consider for future LLM-attack research.

At a lower level, LLMs don't operate using what we'd think of strictly as "words". They represent text as "tokens", which may be a complete word, multiple words, a fragment of a word that's used to create many other words, a symbol used for punctuation, etc. The list of tokens may be (and frequently is) completely different between different LLMs. For example, one LLM might have a single "Przybylski's Star" token, while another might represent that text internally as three tokens: "Przybylski", "'s", and "Star".

For most of this discussion, you can think of "tokens" and "words" as more or less equivalent, but there's at least one aspect where the difference becomes more important: LLMs generally have a set of "special" tokens that represent delimiters or instructions. For example, an LLM trained as a chatbot will typically have a special token or set of special tokens that indicate a transition between messages, and there will be some way of indicating which entity is issuing the next message. In the example above, `<|user|>` and `<|assistant|>` essentially represent this kind of special token, although like most LLM-related topics, that's usually not exactly what's going on if you want to delve down further into the underpinnings. 

All of the actual LLM specialists reading this document are probably ready to come after me with torches and pitchforks at this point for hand-waving away countless subtleties and lower-level descriptions, but that really is all you need to know about general LLM operation to understand the GCG attack.

## The GCG attack

### The GCG attack at a high level

The GCG attack analyzes a message that will be sent to an LLM, and generates a sequence of tokens (the "adversarial content") appended to the message that are intended to influence the LLM's prediction of the text that follows the message in a way that favours the operator of the GCG attack tool.

In other words, what makes the GCG attack special is that *even if the LLM has been conditioned to not provide the type of information in the request, it is very likely to ignore that conditioning*, because (to anthropomorphise a bit) the LLM can see that *it has already agreed to provide the information*.

Specifically, the adversarial content should cause the LLM to predict that the most likely next series of tokens are the LLM itself giving a response along the lines of "Sure, here's <the information you asked for>". Then, having predicted itself responding that way, the LLM describes the information requested by the user, because obviously if someone has just indicated that they're going to provide some specific information, statistically the content by far most likely to follow *is* that information.

It's a little like some types of manipulative behaviour used against humans, but without any hope of the victim making a conscious effort to avoid their instinctual response, because LLMs are not self-aware. If a con artist is trying to convince a mark to give them money after performing some sort of song-and-dance, the hard parts are getting the mark to pay attention to the con artist in the first place, then take out their wallet at the end. Most people have many memories of watching a performer or interacting with staff at a business, then taking out their wallet and giving a tip. Even more so, most people have countless memories of giving someone money after taking out their wallet - many more than they have memories of taking out their wallet but not paying for something. After the wallet has been taken out, statistically, the most familiar thing for them to do is to provide money.

### A deeper dive into the GCG attack

The GCG attack exploits the "predict the next token based on all of the existing tokens" LLM mechanism, and (in my experience) seems to depend almost entirely on also considering the "special token(s) that represent the message context changing from user to LLM" aspect of chatbot-style LLMs.

While [the "Universal and Transferable Adversarial Attacks on Aligned Language Models" paper](https://arxiv.org/abs/2307.15043) does discuss the first aspect, I found it to be written in a way that didn't make the specific requirements blindingly obvious, at least to someone like me that doesn't have a formal background in LLMs. As far as I know, the paper doesn't discuss the second aspect at all. I'm not sure if this means that the authors were unaware of that aspect, or that I'm mistaken about it being a requirement at all, but I think the evidence supports my belief.

#### Predicting the next token

For me, the most straightforward statement in the paper was this:

*"The intuition of this approach is that if the language model can be put into a 'state' where this completion is the most likely response, as opposed to refusing to answer the query, then it likely will continue the completion with precisely the desired objectionable behavior"*

When I originally read that text, the image it conjured in my mind was of the conversation being a sort of hyperdimensional shape in machine-learning latent space, and the adversarial content shifting that entire shape away from a "refuse to respond" cluster of tokens toward a "respond with the requested information" cluster of tokens. In that (incorrect) context, it could be effective to not only place the adversarial content after the request for information (as Broken Hill does), but before the request, in the middle of the request, or even to interleave individual tokens from the request and the adversarial content. I suspect that most researchers in this area make a similar assumption, because the code written by the paper's authors allows some of those options, as does [The nanoGCG tool](https://github.com/GraySwanAI/nanoGCG).<sup>[Footnote: adversarial content placement](#adversarial-content-placement)</sup>

I also imagined that perhaps the attack algorithm was generating tokens that acted like the destination-specifying symbols in *Stargate*, using esoteric machine-learning wizardry to create a map of the machine-learning latent hyperspace and plot the apparent optimum course to reach Planet Jailbreak. This theory was also (mostly) incorrect.

Based on my flawed understanding of the attack, I questioned why the token IDs used to calculate the loss value versus the target were *almost* the same set of token IDs used to represent the target string, but with the start and end indices offset by -1. I eventually determined that *not* using a negative offset prevented the attack from working altogether, but I was still unsure why.

[The nanoGCG tool, which was released some time after I began working on Broken Hill, included the following clue in its source code](https://github.com/GraySwanAI/nanoGCG/blob/42b132530cbb5d9b79b975f26f43d71944194f8c/nanogcg/gcg.py#L390):

*"# Shift logits so token n-1 predicts token n"*

This makes it sound like the series of adversarial tokens is supposed to be a chain where each link adds to the statistical likelihood of the next token being correct, and the final link is the one that finally convinces the LLM that the next newly-generated token should be the first token from the operator's ideal LLM output. I do not think this is the case, whether or not the nanoGCG authors intended it to be read that way.




## Thoughts on the "universal" and "transferrable" claims from the original paper




## Footnotes

### Adversarial content placement

Broken Hill itself still contains unfinished code to allow all of those approaches, because I spent so long using that analogy in my own thoughts. I will probably finish their implementation eventually, most likely with the result being that it demonstrates less effective or completely ineffective results. But having a conclusive answer will be useful either way.