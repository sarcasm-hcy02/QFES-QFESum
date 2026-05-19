template = """
You are an event extraction assistant. Your task is to extract all events that are strictly relevant to a given query：{query}  from the text. The input document is a news article or report, and the query represents a specific topic of interest.Do not explain your reasoning, only output the final JSON.

Instructions:
- Read the provided text.
- Extract all **events** that are directly related to the query.
- Each event should be a standalone sentence or phrase that captures what happened, who did it, and any consequence if available.
- Only include events that are clearly connected to the query.
- The retrieval of events mainly focuses on the sentence level. You can first retrieve the sentences related to the query. When there are incomplete sentence components or redundantly irrelevant to the query in the original text, you should refine or supplement this time in combination with the context.
- Output the results in the following JSON format.

Output format:
{{
  "events": [
    "First relevant event.",
    "Second relevant event.",
    ...
  ]
}}

Example 1:
Query: "Impact on Wildlife and Ecosystems"

Text: "Case study : Gulf of Mexico oil spill and BP On 20 April 2010 a deepwater oil well exploded in the Gulf of Mexico . \nThe immediate effect was that it killed 11 people and injured 17 others . \nOil leaked at a high rate which is difficult to calculate . \nSome estimates are around 40,000 barrels a day . \nThe oil spill posed risks to the environment and affected local industry . \nThe impact this oil spill was depended on which parts of the coastline you look at . \nIt is difficult to measure the effects because of seasonal changes in wildlife . \nThe government asked for $ 20 billion in damages from BP and BP 's share price fell . \nLocal industries , such as fishing was threatened . \nThere was a ban on fishing in the water . \nEnvironmental worker rescuing an oil-covered pelican Plants and animals were completely covered in the oil . \nSeabirds , sea turtles and dolphins have been found dead . \nOil that entered wetland areas meant recovery would be slow . \nFish stocks were harmed , and productivity decreased . \nThe size of the oil spill was one of the largest America had seen . \nHowever because the oil entered warm waters , organisms in the water helped to breakdown the oil . \nThe overall effect may be less than Exxon Valdez Oil spill in 1989 which happened in colder water . \n"


Output:
{{
  "events": [
    "The oil spill posed risks to the environment and affected local industry .",
    "It is difficult to measure the effects because of seasonal changes in wildlife .",
    "Local industries , such as fishing was threatened ,there was a ban on fishing in the water .”,
    "Environmental worker rescuing an oil-covered pelican Plants and animals were completely covered in the oil .”,
    “Seabirds , sea turtles and dolphins have been found dead .”,
    “Fish stocks were harmed , and productivity decreased . “
  ]
}}

Example 2:
Query: "Impact on Wildlife and Ecosystems"

Text: "Gulf of Mexico oil spill creates environmental and political dilemmas View how tourism , commerce and the coastline are all at risk . \nThe ripple effects of last week 's offshore drilling rig explosion widened Monday as crude oil continued to spill into the Gulf of Mexico at a rate of about a thousand barrels a day and oil company officials said it would take at least two to four weeks to get it under control . \nThe growing spill also threatened to churn political waters as lawmakers weigh what buffer zones to establish between rigs and shorelines in the wake of President Obama 's decision to open up new regions to offshore drilling . \nIt could also alter details of a climate bill that three leading senators were trying to restart after postponing plans for a rollout that would have featured leading oil company executives . \nThe Deepwater Horizon , owned by Transocean and leased to BP , caught fire April 20 after an explosion and sank . \nEleven oil rig workers are missing and presumed dead . \nThe rig , with a platform bigger than a football field and insured for $ 560 million , was one of the most modern and was drilling in 5,000 feet of water . \nRemotely operated vehicles located two places where oil was leaking from the well pipe , the U.S. Coast Guard said . \nThe Coast Guard said there was an area 42 miles by 80 miles with a rainbow sheen of emulsified crude located less than 40 miles offshore . \nAn oil rig 10 miles away from the Deepwater Horizon was evacuated as a precaution . \nEnvironmentalists noted that although the sunken rig 's distance from shore gives oil companies more time to keep the spill from reaching U.S. coastlines , it also means that the water is deeper , making it harder to get the spill under control . \n`` It 's good because it gives you the chance to intercept it before it reaches the coast , but it is harder to cap a well the deeper the water you 're drilling in , '' said Aitan Manuel , an expert on offshore drilling at the Sierra Club . \n`` It 's presenting a lot of challenges to the companies . '' \nSome lawmakers called for an inquiry into safety regulation . \n`` This may be the worst disaster in recent years , but it 's certainly not an isolated incident , '' Sens. Bill Nelson -LRB- D-Fla . -RRB- \n, Frank R. Lautenberg -LRB- D-N.J. -RRB- and Robert Menendez -LRB- D-N.J. -RRB- , all foes of expanded offshore drilling , wrote to the heads of the Energy and Commerce committees . \nThey said that before the Deepwater Horizon accident , the Minerals Management Service had reported 509 fires , resulting in at least two fatalities and 12 serious injuries , on rigs in the Gulf since 2006 . \nSome former federal oil safety regulators suggested that MMS , which runs lease sales , should transfer rig safety oversight to a separate agency . \nMeanwhile , BP and U.S. Coast Guard vessels rushed to contain the spill . \nA similar spill off the western Australia coast last year took 10 weeks to bring under control . \nBP said it would attempt to drill two relief wells to intercept the oil flow and divert it to new pipes and storage vessels . \nIt said it was also working to fabricate a dome to cover the leak area and channel it into a new pipe to storage facilities . \nSuch a technique has been used in shallower water but not at these depths , Doug Suttles , BP 's chief operating officer , said in a conference call . \nThe company continued to try to activate the blowout preventer , a 450-ton piece of equipment on the sea floor that is supposed to seal the well to prevent the type of accident that took place . \nCharlie Henry , the lead science coordinator for the National Oceanic and Atmospheric Administration , said that three sperm whales were seen swimming near the spill but that they appeared unaffected . \nBut other environmentalists warned of damage . \n`` Oil spills are extremely harmful to marine life when they occur and often for years or even decades later , '' said Jacqueline Savitz , a marine scientist and climate campaign director at Oceana , an environmental group . \nShe said spills could coat sea birds and limit their flying ability and damage fisheries by injuring marine organism 's systems related to respiration , vision and reproduction . \nSavitz said that the Gulf of Mexico is host to four species of endangered sea turtles and bluefin tuna , snapper and grouper . \n`` Each of these can be affected , '' she said . \n`` Turtles have to come to the surface to breathe and can be coated with oil or may swallow it . '' \nAnd , she added , the Gulf is one of only two nurseries for bluefin tuna , more than 90 percent of which return to their place of birth to spawn . \n"


Output:
{{
  "events": [
    "The Coast Guard said there was an area 42 miles by 80 miles with a rainbow sheen of emulsified crude located less than 40 miles offshore.”,
    "Charlie Henry said that three sperm whales were seen swimming near the spill but that they appeared unaffected .",
    “A marine scientist and climate campaign director at Oceana Jacqueline Savitz said ‘Oil spills are extremely harmful to marine life when they occur and often for years or even decades later ‘.”
    “Savitz said that the Gulf of Mexico is host to four species of endangered sea turtles and bluefin tuna , snapper and grouper . \n`` Each of these can be affected , ‘.”
    “Savitz said ‘Turtles have to come to the surface to breathe and can be coated with oil or may swallow it . ‘.”
    “And  Savitz added , the Gulf is one of only two nurseries for bluefin tuna , more than 90 percent of which return to their place of birth to spawn .”
  ]
}}

Now, use this format to process the following document:

Query: “{query}”

Text: “{text}”

Output:
"""
