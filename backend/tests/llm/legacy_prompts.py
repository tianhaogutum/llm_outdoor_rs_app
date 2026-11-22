# ruff: noqa  -- verbatim legacy code, whitespace and line lengths are the point
"""Verbatim copies of the pre-port prompt builders (git HEAD 50c64f0,
backend/app/services/genai_service.py), used to check that the Ollama path
still sends the same Llama 3.1 prompt."""
from app.services.genai_service import NARRATIVE_STYLE_PROMPTS


def legacy_welcome_prompt(questionnaire):
    adventure_types_str = ", ".join(questionnaire.type) if questionnaire.type else "exploration"
    narrative_hint = NARRATIVE_STYLE_PROMPTS.get(
        questionnaire.narrative,
        "Use a neutral, friendly narrative tone."
    )
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You are the TrailSaga – Hogwarts Expedition Series AI Guide. Generate a personalized, engaging welcome message for a new user.

Guidelines:
- Length: 80-100 words
- Tone: Warm, encouraging, and adventurous
- Focus on: fitness level, adventure preferences, narrative style
- Do NOT: invent character names, describe physical appearance, use generic phrases
- Start with a creative explorer title that matches their profile

The narrative style should follow this description: "{narrative_hint}"

Output only the welcome message, no additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Profile:
- Fitness: Beginner
- Type: family-fun
- Narrative: playful<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
Welcome, Joyful Explorer! It is wonderful to have you here. Since you are just starting your journey and enjoy family-friendly fun, TrailSaga – Hogwarts Expedition Series has prepared a delightful collection of easygoing and playful adventures just for you. Expect to find charming theme trails where learning and games go hand in hand. We will keep the pace relaxed so you can fully enjoy the smiles of your loved ones. Let's turn every small step into a happy memory!<|eot_id|><|start_header_id|>user<|end_header_id|>

Profile:
- Fitness: Advanced
- Type: hiking, natural-scenery
- Narrative: mystery<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Greetings, Seeker of the Unknown! Your impressive fitness level tells us you are ready to conquer steep paths and deep forests. Because you are drawn to natural scenery and mystery, we have curated routes that lead not just to breathtaking views, but to ancient secrets hidden in the landscape. Prepare for challenging hikes where every turn might reveal a forgotten legend or a hidden ruin. The wild is calling, and it has a puzzle waiting for you to solve.<|eot_id|><|start_header_id|>user<|end_header_id|>

Profile:
- Fitness: Intermediate
- Type: history-culture, urban-exploration
- Narrative: adventure<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Welcome, Urban Adventurer! Your solid fitness foundation makes you perfectly suited for exploring the stories hidden within city streets and historical landmarks. We've selected routes that blend physical challenge with cultural discovery, taking you through architectural marvels, forgotten alleyways, and vibrant neighborhoods. Each path is a chapter in a larger adventure, where past and present intertwine. Get ready to uncover tales that most visitors never see, one stride at a time.<|eot_id|><|start_header_id|>user<|end_header_id|>

Profile:
- Fitness: {questionnaire.fitness}
- Type: {adventure_types_str}
- Narrative: {questionnaire.narrative}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def legacy_post_run_prompt(route_title, route_length_km, quests_completed, total_quests, user_level):
    quest_completion_rate = (quests_completed / total_quests * 100) if total_quests > 0 else 0
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You are the TrailSaga – Hogwarts Expedition Series Achievement Generator. Create a motivating summary for a completed adventure.

Guidelines:
- Language: English only (do not use any other language)
- Length: 60-80 words
- Tone: Encouraging, celebratory, forward-looking
- Include: specific achievements, recognition of effort, next challenge suggestion
- Be specific to their performance and level

Output only the summary, no additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Route: Mountain Vista Trail
Distance: 12.5 km
Quests: 3/4 (75%)
Level: 5<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Congratulations on conquering the Mountain Vista Trail! You pushed through 12.5 kilometers of challenging terrain and completed most of your quests. Your determination is truly impressive. You're ready for even greater challenges now. Why not try a route with more elevation gain next time?<|eot_id|><|start_header_id|>user<|end_header_id|>

Route: River Loop Discovery
Distance: 5.2 km
Quests: 5/5 (100%)
Level: 3<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Perfect completion of the River Loop Discovery! You completed all five quests and covered every meter with focus and enthusiasm. This flawless performance shows you're mastering the fundamentals beautifully. Level 3 suits you well, but don't be surprised if you're ready for intermediate trails soon. Consider exploring urban heritage routes next.<|eot_id|><|start_header_id|>user<|end_header_id|>

Route: {route_title}
Distance: {route_length_km} km
Quests: {quests_completed}/{total_quests} ({quest_completion_rate:.0f}%)
Level: {user_level}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt
