These strategies define how the Manager/Synthesizer component (often the final agent in a `refine_last` style team, or a dedicated step in a more complex workflow) combines inputs from preceding agents to generate imaginative prompts.

**Proposed Creative Synthesis Strategies for Prompt Generation:**

1.  **Metaphorical Synthesis:**
    *   **Team (Conceptual):** [Aspect Analyzer A, Aspect Analyzer B, ..., **Metaphorical Synthesizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `metaphorical_synthesis`. Requires specific prompting for the final agent.
    *   **Mechanics:** Initial agents analyze text input or image aspects. The final agent receives these analyses and is explicitly prompted to synthesize them *using a provided or generated metaphor* (e.g., "Combine the elements described in the context and express them through the metaphor of a 'clockwork jungle'").
    *   **Use Case (Prompt Generation):** Generating abstract, non-literal, or highly interpretive prompts. Excellent for breaking creative blocks or exploring specific conceptual themes visually.
    *   **Pros:** High creative potential, generates unique conceptual angles, encourages unexpected visual connections.
    *   **Cons:** Results are highly interpretive and depend on the LLM's ability to handle metaphor effectively. Less direct control over specific output elements compared to descriptive synthesis.

2.  **Conceptual Blending:**
    *   **Team (Conceptual):** [Concept Extractor A, Concept Extractor B, ..., **Concept Blender (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `conceptual_blend`. Requires specific prompting for the final agent.
    *   **Mechanics:** Initial agents identify key concepts, objects, or styles. The final agent is prompted to *fuse* these disparate concepts into a description of a *single, novel hybrid entity or scene*.
    *   **Use Case (Prompt Generation):** Intentionally creating unique, surreal, or previously non-existent subjects/objects/worlds for the image generation model. Strong tool for ideation.
    *   **Pros:** Excellent for generating genuine novelty and originality. Directly combines diverse inputs into something new.
    *   **Cons:** Can be unpredictable; success depends on the semantic "distance" of concepts being blended and the LLM's ability to fuse them coherently. Risk of nonsensical output if not guided.

3.  **Stylistic Mashup / Textual Style Transfer:**
    *   **Team (Conceptual):** [Content Analyzer A, Content Analyzer B, ..., **Stylizer Agent (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `stylistic_mashup`. Requires specific prompting and potentially style examples/keywords for the final agent.
    *   **Mechanics:** Initial agents focus on extracting the *content* (objects, actions, scene). The final agent synthesizes this content but is explicitly instructed to render the entire prompt text *in a specific, often unrelated, literary or textual style* (e.g., "Write a prompt containing [synthesized content] in the style of a 1920s newspaper report / a pirate's sea shanty / Python code comments").
    *   **Use Case (Prompt Generation):** Injecting strong, specific textual flavors into the prompt itself, influencing the mood, interpretation, and potentially the visual style generated. Creating juxtapositions.
    *   **Pros:** Powerful control over prompt's textual tone and style. Can lead to unique interpretations by the image generation model based on the stylistic framing.
    *   **Cons:** Effectiveness depends heavily on the LLM's ability to mimic the target style accurately. Chosen style might clash awkwardly or confusingly with the core content.

4.  **Dialectical Amplification:**
    *   **Team (Conceptual):** [Contrasting Perspective A, Contrasting Perspective B, ..., **Tension Synthesizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `dialectical_amplification`. Requires specific prompting for the final agent.
    *   **Mechanics:** Initial agents are designed to provide opposing or contrasting viewpoints on the input (e.g., Agent A focuses on 'order', Agent B on 'chaos'). The final agent is prompted to *highlight and exaggerate the conflict, tension, or paradox* between these inputs in the final prompt.
    *   **Use Case (Prompt Generation):** Creating prompts that emphasize visual drama, dynamic contrast, opposing forces, or unresolved conceptual tensions.
    *   **Pros:** Generates prompts with inherent energy and visual interest derived from conflict. Explores conceptual edges.
    *   **Cons:** Might require carefully designed initial agent roles to ensure meaningful contrast. Can sometimes oversimplify nuances into binary oppositions.

5.  **Stochastic Excerpts (Cut-Up):**
    *   **Team (Conceptual):** [Text Generator A, Text Generator B, ..., **Fragment Assembler (Logic/LLM)**]
    *   **Strategy Type (Internal):** *Custom Logic* potentially mixed with LLM calls. Less likely to be a pure `refine_last`. Might need Python code implementing the sampling.
    *   **Mechanics:** Programmatically (or via complex LLM instruction) selects random words, phrases, or sentences from the outputs of preceding agents. These fragments are then assembled into a prompt, perhaps with minimal LLM-generated connective text or just joined raw.
    *   **Use Case (Prompt Generation):** Intentionally breaking logical flow and semantic coherence to achieve serendipitous, unexpected, or chaotic results. Useful for experimental art or overcoming creative ruts.
    *   **Pros:** Can produce highly unique and surprising combinations completely outside conventional thinking. Good for generating "happy accidents."
    *   **Cons:** Highly unpredictable and uncontrollable. High probability of generating unusable or nonsensical prompts. Requires careful tuning of sampling parameters.

6.  **Sensory Translation Synthesis:**
    *   **Team (Conceptual):** [Aspect Analyzer A, Aspect Analyzer B, ..., **Synesthetic Translator (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `sensory_translation`. Requires specific prompting for the final agent.
    *   **Mechanics:** Initial agents analyze various aspects of the input. The final agent is prompted to describe the *combined essence* of these inputs as if experienced through a *different sensory modality* (e.g., "Describe the 'sound' of this scene," "What is the 'texture' of this concept?").
    *   **Use Case (Prompt Generation):** Creating extremely abstract, metaphorical, and synesthetic prompts that challenge the image generator's interpretation.
    *   **Pros:** Pushes creative boundaries significantly. Generates unique sensory language rarely used in standard prompts.
    *   **Cons:** Highly abstract; the resulting image's connection to the prompt might be tenuous or unpredictable. Success depends heavily on both the LLM's and the image generator's interpretive abilities.

7.  **Instructional Inversion:**
    *   **Team (Conceptual):** [Feature Extractor A, Feature Extractor B, ..., **Inversion Agent (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `instructional_inversion`. Requires specific prompting for the final agent.
    *   **Mechanics:** Initial agents identify key features, concepts, or objects. The final agent synthesizes these core ideas and is then prompted to generate a prompt describing the *conceptual opposite*, the *absence* of these elements, or a scene defined *by their inversion*.
    *   **Use Case (Prompt Generation):** Exploring themes through negation or absence. Creating prompts focused on negative space, contrast, or minimalism. Generating counter-intuitive scenarios.
    *   **Pros:** Novel approach to prompt definition. Can lead to visually interesting compositions focusing on what *isn't* there.
    *   **Cons:** Defining the "opposite" can be ambiguous for complex concepts. Requires careful prompting of the final agent to avoid simple negations (e.g., "not blue" vs. "the color that is the absence of blue").


**Proposed Creative Synthesis Strategies for Prompt Generation (Continued):**

8.  **Persona-Based Synthesis:**
    *   **Team (Conceptual):** [Aspect Analyzer A, Aspect Analyzer B, ..., **Persona Interpreter (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `persona_synthesis`. Requires specific prompting defining the persona for the final agent.
    *   **Mechanics:** Initial agents extract features, details, or concepts. The final agent receives this context and is prompted to synthesize and describe it *from the perspective of a specific character or archetype* (e.g., "Describe [agent inputs] as perceived by a cynical detective," "Synthesize this information into a prompt from the viewpoint of an excited child," "How would a thousand-year-old tree describe this?"). The persona could be predefined or dynamically provided.
    *   **Use Case (Prompt Generation):** Injecting strong character voice, specific biases, or unique observational filters into the prompt. Useful for creating images that reflect a particular subjective viewpoint or narrative framing.
    *   **Pros:** Creates prompts with distinct personality and perspective. Can subtly influence details, focus, and emotional tone based on the chosen persona.
    *   **Cons:** Requires well-defined personas. Effectiveness depends on the LLM's ability to adopt and consistently apply the persona's viewpoint. Can result in overly niche or specific prompts if the persona is too obscure.

9.  **Temporal Projection / Evolution:**
    *   **Team (Conceptual):** [Current State Analyzer A, Current State Analyzer B, ..., **Temporal Projector (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `temporal_projection`. Requires specific prompting defining the time shift (past/future) and potentially the nature of change (decay, growth, technological advance).
    *   **Mechanics:** Initial agents describe the subject as it is presented (in the input image or text). The final agent synthesizes this current state and is prompted to extrapolate and describe how the subject might appear *in a specified future or past*, potentially incorporating specific change factors (e.g., "Based on [agent inputs], describe this scene 1000 years in the future, showing signs of technological evolution," "Describe the ancient ruins from which this object might have originated," "Show this character weathered by decades of hardship").
    *   **Use Case (Prompt Generation):** Exploring concepts of time, decay, growth, history, or futurism. Generating variations of a subject across different eras. Creating prompts for historical fiction, sci-fi, or fantasy settings.
    *   **Pros:** Introduces a dynamic element of time and change. Generates imaginative variations based on temporal logic.
    *   **Cons:** Relies on the LLM's ability to logically (or creatively) extrapolate changes over time. Can be speculative and might require careful prompting to guide the type of temporal transformation desired.

10. **Functional Reinterpretation / Exaptation:**
    *   **Team (Conceptual):** [Object/Scene Describer A, Structure/Form Analyzer B, ..., **Functional Reinterpreter (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `functional_reinterpretation`. Requires specific prompting suggesting or demanding a new function.
    *   **Mechanics:** Initial agents describe the object's or scene's appearance, components, and perhaps intended function. The final agent synthesizes this description and is prompted to reimagine and describe the subject as if it were *adapted or repurposed for a completely different, often unexpected, function* (e.g., "Describe how the 'futuristic car' from the input could be repurposed as a 'deep-sea exploration vehicle'," "Reimagine this 'serene landscape' as a 'battleground map'," "Describe this 'delicate teacup' adapted into a 'brutal weapon'").
    *   **Use Case (Prompt Generation):** Generating creative designs based on repurposing existing forms (exaptation). Creating prompts that juxtapose form and function. Exploring "what if" design scenarios.
    *   **Pros:** Spurs creativity by breaking assumptions about function. Can lead to highly original hybrid designs or scenes. Encourages thinking about form independent of original purpose.
    *   **Cons:** Requires a plausible (or creatively implausible) link between the original form and the new function. Success depends on the LLM's ability to imaginatively bridge the gap.

11. **Emotional Resonance Mapping:**
    *   **Team (Conceptual):** [Emotional Tone Analyzer A, Symbolic Element Reader B, ..., **Emotion Visualizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `emotional_resonance`. Requires specific prompting focusing on emotional translation.
    *   **Mechanics:** Initial agents focus on identifying the emotional tones, moods, symbolism, or psychological impact conveyed by the input. The final agent synthesizes these *emotional* aspects and is prompted to translate them into a *visual description* that evokes those specific feelings, potentially deprioritizing literal object representation (e.g., "Based on the feelings of 'melancholy' and 'nostalgia' detected, describe a scene that visually embodies these emotions," "Translate the 'aggressive energy' and 'sharp angles' described into a visual prompt").
    *   **Use Case (Prompt Generation):** Creating highly atmospheric, mood-driven, or abstract prompts where emotional impact is the primary goal. Useful for abstract art, character emotion studies, or evocative environments.
    *   **Pros:** Focuses directly on translating subjective feeling into visual language. Can generate powerful, mood-focused imagery.
    *   **Cons:** Highly abstract and interpretive. The link between the generated visual and the original input might be tenuous. Relies heavily on the LLM's and GenAI's ability to map emotions to visual elements.


**Proposed Creative Synthesis Strategies for Prompt Generation (Continued):**

12. **Design Process Evocation (Reworked Chain-of-Thought):**
    *   **Team (Conceptual):** [Material Analyst, Form Analyst, Constraint Analyzer, ..., **Design Process Synthesizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `design_process_synthesis`. Requires specific prompting for the final agent to visualize the *process*, not just the outcome.
    *   **Mechanics:** Initial agents break down the design problem (from user prompt) or analyze an inspirational image. The final agent synthesizes these requirements/observations and generates a prompt describing *the act or artifacts of designing* the subject, rather than the finished piece. It evokes iteration, sketching, annotation, material exploration, etc. Example prompt target: "Workbench showing iterative design sketches for an Art Deco inspired teapot, ceramic material samples nearby, blueprint fragments, soft lighting."
    *   **Use Case (Prompt Generation):** Generating images that visualize creation, concept development, or the design process itself. Ideal for concept art, illustrating workflow, or adding a meta-layer to the subject.
    *   **Pros:** Creates unique visuals focusing on process, not just product. Can incorporate multiple design elements (sketches, notes, materials) naturally. Offers a meta-narrative about creation.
    *   **Cons:** Needs careful prompting of the synthesizer agent. Image generator must understand terms like 'sketch', 'blueprint', 'iteration', 'mood board'. May obscure the final design if not balanced.

13. **Parameterized Abstraction / Deconstruction:**
    *   **Team (Conceptual):** [Key Element Extractor A, Structural Analyzer B, ..., **Abstractor/Deconstructor (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `parameterized_abstraction`. Prompting for the final agent needs to specify the *style* and *degree* of abstraction/deconstruction.
    *   **Mechanics:** Initial agents identify core shapes, structures, colors, or themes. The final agent synthesizes these and generates a prompt describing the subject in a *highly abstracted or deconstructed manner*, guided by parameters. Examples: "Minimalist essence of [subject] using only primary colors and geometric shapes," "Deconstructed [subject], its core components floating in a void," "Wireframe schematic diagram highlighting the structural form of [subject]."
    *   **Use Case (Prompt Generation):** Generating abstract art, minimalist graphics, technical illustrations, or exploring the fundamental structure of a subject visually. Useful for analysis or stylistic simplification.
    *   **Pros:** Focuses on core forms and concepts. Can produce clean, modern, or analytical aesthetics. Controllable degree of simplification.
    *   **Cons:** Detail is inherently lost. Requires clear instructions for the abstraction style. Image generator's interpretation of abstract concepts is key.

14. **Cross-Modal Translation (Beyond Sensory):**
    *   **Team (Conceptual):** [Concept Analyzer A, Mood Analyzer B, Structure Analyzer C, ..., **Cross-Modal Translator (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `cross_modal_translation`. Prompting must define the target non-visual domain (music, math, data, language).
    *   **Mechanics:** Initial agents extract abstract concepts, structures, emotions, or patterns. The final agent synthesizes these and generates a prompt describing a *visual representation of that information as if translated into another domain*. Examples: "Visual representation of [subject's core rhythm/pattern] as musical score fragments," "Fractal geometry pattern derived from the structural complexity of [subject]," "Typographic art landscape formed by the key themes of [subject]."
    *   **Use Case (Prompt Generation):** Highly experimental art. Visualizing abstract data or concepts from non-visual fields. Creating unique textures or patterns based on different logical systems. Pushing interpretive boundaries.
    *   **Pros:** Extremely novel conceptual approach. Bridges different domains creatively. Potential for unique data-driven or structurally abstract visuals.
    *   **Cons:** Very abstract and unpredictable. Success hinges heavily on the interpretive capabilities of both the synthesizing LLM and the image generator. High risk of visually incoherent or irrelevant results.

15. **Rule-Based Mutation / Algorithmic Transformation:**
    *   **Team (Conceptual):** [Component Identifier A, Parameter Extractor B, ..., **Mutator Agent (Final Agent/Logic)**]
    *   **Strategy Type (Internal):** `refine_last` with complex rule instructions, or potentially custom logic (`algorithmic_mutation`). Rules defined in the prompt or parameters.
    *   **Mechanics:** Initial agents identify components, properties, or parameters. The final synthesis step involves applying *explicit transformation rules* to the description. Rules can be simple ("Replace all sharp corners with rounded edges," "Intensify color saturation by 50%") or complex descriptive analogs of algorithms ("Apply a 'glitch art' aesthetic with datamoshing effects," "Describe a 'recursive cellular automaton' pattern evolving across the surface of [subject]").
    *   **Use Case (Prompt Generation):** Simulating algorithmic or generative art processes via descriptive prompts. Applying specific visual effects or systematic transformations. Exploring procedural generation concepts textually.
    *   **Pros:** Allows systematic, rule-driven transformations. Can create aesthetics associated with generative art. More controlled than purely random methods if rules are well-defined.
    *   **Cons:** Requires translating algorithmic ideas into descriptive language the image generator can understand. Effectiveness depends heavily on the GenAI's interpretation of effect descriptions ("glitch art," "recursive"). Complex rules can be hard to formulate in prompts.

16. **Narrative Context Injection:**
    *   **Team (Conceptual):** [Subject Describer A, Style Analyzer B, ..., **Narrative Contextualizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `narrative_context`. Requires prompt input specifying the desired narrative setting, action, or mood.
    *   **Mechanics:** Initial agents describe the core subject and style. The final agent synthesizes this information and is prompted to *embed the subject within a richer narrative context*, adding background, implied action, environmental details, or story cues. Examples: "The [subject described by agents] resting on a workbench in a cluttered, dimly lit inventor's workshop," "A lone [subject] sits abandoned in a post-apocalyptic wasteland under a stormy sky."
    *   **Use Case (Prompt Generation):** Creating full scenes, establishing mood and atmosphere, storytelling, grounding abstract subjects, providing context that influences lighting and composition.
    *   **Pros:** Adds significant depth, atmosphere, and storytelling potential. Makes prompts more specific and evocative. Guides the image generator towards a complete scene.
    *   **Cons:** The injected context might overshadow the original subject. Requires creative input for the narrative setting. Relies on the LLM's ability to blend the subject seamlessly into the scene.

These strategies, integrated into ArtAgents' team/workflow system, would provide users with a powerful suite of tools for moving beyond simple description towards highly creative, transformative, and conceptually driven prompt engineering, fully leveraging the potential of multi-agent collaboration for artistic exploration.

**Conceptually Different Synthesis Strategies for Prompt Generation:**

17. **Knowledge-Augmented Synthesis:**
    *   **Team (Conceptual):** [Topic Extractor A, Style Analyzer B, ..., **Knowledge Integrator (Final Agent/Logic)**]
    *   **Strategy Type (Internal):** Requires interaction with external knowledge sources (e.g., web search API, vector database lookup) during synthesis, likely custom logic + LLM call. `knowledge_augmented_synthesis`.
    *   **Mechanics:** Initial agents identify key subjects, themes, or entities. The synthesis step *actively queries an external knowledge source* for relevant facts, context, variations, or related concepts based on the agent inputs. The final agent then weaves both the agent perspectives *and* the retrieved external knowledge into the prompt. Example: Agents describe a "griffin." Synthesizer fetches mythological variations or biological details of eagles/lions and integrates them: "A griffin combining [Agent A's description] with the scale texture of a pangolin and talons adapted for arctic ice gripping [external knowledge]."
    *   **Use Case (Prompt Generation):** Grounding fantastical concepts with real-world details, adding specific verified information, discovering unexpected connections, enhancing realism or specificity by incorporating external data beyond the agents' initial scope.
    *   **Pros:** Introduces objective or novel information from external sources. Can increase depth, accuracy (if desired), or creative variation based on real-world data. Bridges LLM knowledge with curated external databases.
    *   **Cons:** Requires external API/database integration. Adds complexity and potential latency. Retrieved knowledge might clash with creative intent if not carefully filtered or prompted. Risk of information overload in the prompt.

18. **Constraint Optimization Synthesis:**
    *   **Team (Conceptual):** [Feature Requester A, Constraint Definer B, ..., **Constraint Solver Synthesizer (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `constraint_optimization_synthesis`. Requires careful prompting of the final agent to prioritize meeting constraints.
    *   **Mechanics:** Agents define desired features *and* explicit constraints (e.g., "must be symmetrical," "low polygon count aesthetic," "uses only materials found in a desert," "maximum visual complexity level: low"). The final agent attempts to synthesize a prompt describing a subject or scene that *satisfies all specified constraints* while incorporating the desired features. The prompt might even articulate how the constraints are met.
    *   **Use Case (Prompt Generation):** Design under constraints, exploring solutions to visual problems, generating assets for specific technical limitations (e.g., game development), ensuring adherence to specific rules or guidelines.
    *   **Pros:** Directly addresses problem-solving in design. Can lead to elegant or efficient solutions driven by limitations. Useful for practical design tasks where rules must be followed.
    *   **Cons:** Effectiveness depends heavily on the LLM's ability to understand and adhere to constraints expressed in natural language. Complex or conflicting constraints might be impossible to satisfy or lead to nonsensical results.

19. **Audience-Targeted Synthesis:**
    *   **Team (Conceptual):** [Content Analyzer A, Style Analyzer B, ..., **Audience Adapter (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `audience_targeted_synthesis`. Requires defining the target audience in the prompt to the final agent.
    *   **Mechanics:** Initial agents analyze the core content and style. The final agent synthesizes this information but *tailors the descriptive language, complexity, focus, and potentially even the implied emotional tone* specifically for a designated audience (e.g., "Describe [agent inputs] in a way that would excite a young child," "Generate a prompt explaining [agent inputs] visually for a scientific journal," "Create a prompt about [agent inputs] designed to evoke nostalgia in someone from the 1980s").
    *   **Use Case (Prompt Generation):** Creating visuals for specific demographics or communication goals. Tailoring marketing imagery. Exploring how viewpoint changes based on the intended receiver. Educational content generation.
    *   **Pros:** Creates prompts (and potentially images) fine-tuned for specific communication impact. Introduces empathy and user-centered design principles into prompt engineering.
    *   **Cons:** Defining audience characteristics effectively in a prompt can be challenging. Relies on the LLM having good models of different audience types and how to communicate with them visually/textually.

20. **Resource-Constrained Synthesis (Material/Component Limitation):**
    *   **Team (Conceptual):** [Core Concept Agent A, Style Agent B, ..., **Resource Limiter (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `resource_constrained_synthesis`. Requires specifying the resource limitations (materials, components, color palette, keyword count) in the prompt to the final agent.
    *   **Mechanics:** Agents define the core idea or subject. The final agent synthesizes this concept but generates a prompt describing it as if *constructed only from a severely limited set of specified resources* or adhering to a strict limit (e.g., "Describe [subject] built entirely from driftwood, rusty nails, and twine," "Generate a prompt for [subject] using only 5 essential keywords," "Depict [scene] using only a two-color palette: black and yellow ochre").
    *   **Use Case (Prompt Generation):** Simulating low-resource scenarios, exploring minimalist aesthetics, challenging creativity through limitation, generating assets with specific material or palette constraints.
    *   **Pros:** Forces creative problem-solving and simplification. Can lead to unique aesthetics defined by scarcity. Good for exploring specific material textures or color theories.
    *   **Cons:** The limitation might make it impossible to represent the core concept adequately. Requires the image generator to interpret the resource constraints accurately.

21. **Constructive Synthesis (Assembly Logic):**
    *   **Team (Conceptual):** [Component Identifier A, Material Specifier B, Form Analyzer C, ..., **Assembler Agent (Final Agent)**]
    *   **Strategy Type (Internal):** `refine_last` or dedicated `constructive_synthesis`. Requires prompting the final agent to focus on assembly/structure.
    *   **Mechanics:** Agents identify parts, materials, and overall shape. The final agent synthesizes this information into a prompt that describes the subject *in terms of its construction, assembly, or layering*. It focuses on *how* parts connect, how materials are joined, or the underlying structure. Examples: "An exploded diagram view showing the assembly of [subject]," "Layer-by-layer construction of [subject], revealing internal structure," "Close-up showing intricate joinery and material transitions on [subject]."
    *   **Use Case (Prompt Generation):** Generating technical illustrations, blueprints (stylized), cutaways, exploded views, or images emphasizing craftsmanship and structure. Visualizing how something is put together.
    *   **Pros:** Provides insight into structure and assembly. Creates visually distinct technical or analytical imagery. Focuses on the "how" rather than just the "what."
    *   **Cons:** May require the image generator to understand technical terms (exploded view, joinery, layering). Might de-emphasize the overall aesthetic for structural detail.

These strategies move beyond simple content merging or stylistic changes, focusing instead on integrating external knowledge, solving problems under constraints, considering the audience, imposing limitations, or revealing construction logic – offering conceptually distinct pathways for creative exploration in ArtAgents.