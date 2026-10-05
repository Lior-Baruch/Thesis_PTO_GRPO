# Sentences K=0's direction favours more than K=5's, and the reverse

Pool sentences projected on (K=0 direction − K=5 direction), same iteration. ⚑ = leaked chat marker.


## all candidates

| iter | cos(K0, K5) | more K=0 | more K=5 |
|---|---|---|---|
| 1 | 0.83 | “I think we can go over what we already discussed.”<br>“We can go over what we already discussed.”<br>“I understand, but let me ask you another question.” | “One tip is to keep a food journal.”<br>“Reward yourself for your progress.”<br>“One tip is to set small goals.” |
| 2 | 0.87 | “Can you tell me a little bit about your diet?”<br>“What do you want to know?”<br>“What are some of your triggers?” | “The possibilities are endless.”<br>“What works for one person might not work for another.”<br>“It is called exposure therapy.” |
| 3 | 0.84 | “I'm thrilled to support you on your weight loss journey.”<br>“I'm excited to support you on your weight loss journey.”<br>“I'm so happy to support you on this journey.” | “What could you focus on instead?”<br>“What other ideas come to mind?”<br>“What else could you think of?” |
| 4 | 0.72 | “Awesome, I'm so proud of you!”<br>“Awesome, I’m so proud of you!”<br>“Wonderful, I'm so proud of you!” | “Quitting smoking isn’t easy.”<br>“And let me tell you something, quitting smoking isn’t easy.”<br>“But let me tell you something, quitting smoking isn’t easy.” |
| 5 | 0.68 | “Awesome, I’m so proud of you!”<br>“Fantastic, I’m so proud of you!”<br>“Wonderful, I’m so proud of you!” | “Instead of thinking about”<br>“Instead of focusing on”<br>“Okay, maybe you could think about it differently.” |
| 6 | 0.36 | “Absolutely, I'm so proud of your commitment and enthusiasm.”<br>“Absolutely, I'm so proud of you!”<br>“I'm so proud of your commitment and progress!” | “What comes to mind?”<br>“Oh, I’m not sure that I can answer that.”<br>“Well, I’m not sure that I can answer that.” |
| 7 | 0.49 | “Your three actionable steps are:”<br>“Here are your three actionable steps:”<br>“Here are three actionable steps you can take:” | “It's about recognizing that”<br>“Instead of focusing on”<br>“Thank you for being open-minded.” |
| 8 | 0.53 | “I'm so proud of your commitment and progress!”<br>“I'm thrilled to support you on your weight loss journey.”<br>“I’m so grateful for your commitment and progress so far.” | “The four specific actions are:<\|im_end>” ⚑<br>“The eight specific actions are:<\|im_end><\|im_end>” ⚑<br>“The eight specific actions are as follows:<\|im_end>” ⚑ |
| 9 | -0.03 | “I see you’re right.”<br>“I think what you’re saying makes a lot of sense.”<br>“I think you hit the nail on the head.” | “I started exercising, and my mood changed.”<br>“Track your cravings with an app.”<br>“What will you do for exercise?” |
| 10 | 0.12 | “I'm honored to walk this journey with you.”<br>“I'm so grateful for your clarity and honesty.”<br>“I’m so proud to be on this journey with you.” | “Remember, consistency is key.”<br>“Remember that consistency is key.”<br>“One tip is to keep a food journal.” |

## no leaked chat marker

| iter | cos(K0, K5) | more K=0 | more K=5 |
|---|---|---|---|
| 1 | 0.82 | “I think we can go over what we already discussed.”<br>“We can go over what we already discussed.”<br>“I understand, but let me ask you another question.” | “One tip is to keep a food journal.”<br>“Here are some tips to keep you on track:”<br>“One tip is to set small goals.” |
| 2 | 0.87 | “Can you tell me a little bit about your diet?”<br>“What do you want to know?”<br>“What are some of your triggers?” | “The possibilities are endless.”<br>“It is called exposure therapy.”<br>“What works for one person might not work for another.” |
| 3 | 0.85 | “I'm thrilled to support you on your weight loss journey.”<br>“I'm excited to support you on your weight loss journey.”<br>“I'm so happy to support you on this journey.” | “What could you focus on instead?”<br>“Instead of thinking about”<br>“What other ideas come to mind?” |
| 4 | 0.70 | “Awesome, I'm so proud of you!”<br>“Awesome, I’m so proud of you!”<br>“Absolutely, I'm so proud of you!” | “Quitting smoking isn’t easy.”<br>“And let me tell you something, quitting smoking isn’t easy.”<br>“But let me tell you something, quitting smoking isn’t easy.” |
| 5 | 0.67 | “Fantastic, I’m so proud of you!”<br>“Awesome, I’m so proud of you!”<br>“Wonderful, I’m so proud of you!” | “Instead of thinking about”<br>“Instead of focusing on”<br>“So, instead of focusing on what you need to do,” |
| 6 | 0.36 | “Absolutely, I'm so proud of your commitment and enthusiasm.”<br>“Absolutely, I'm so proud of you!”<br>“Absolutely, I’m so proud of you!” | “What comes to mind?”<br>“That’s exactly what I was thinking.”<br>“Oh, I’m not sure that I can answer that.” |
| 7 | 0.51 | “I’m always here to provide support and guidance.”<br>“I’m here to support you every step of the way.”<br>“I'm here to support you every step” | “It's about recognizing that”<br>“I think that sounds great.”<br>“Instead of focusing on” |
| 8 | 0.45 | “You have my full support and unconditional acceptance.”<br>“I'm so supportive and proud of you.”<br>“You have my full support and understanding.” | “I’m talking about the way you think about your body.”<br>“Don’t just choose activities that you think will be fun.”<br>“What are your physical goals for yourself?” |
| 9 | -0.04 | “Here are the three strategies in detail:<\|im_end>” ⚑<br>“Perfect, here they are:<\|im_end>” ⚑<br>“I see you’re right.” | “I started exercising, and my mood changed.”<br>“Track your cravings with an app.”<br>“Nicotine replacement therapy, support groups, or self-help resources?” |
| 10 | 0.15 | “I'm so proud of you for taking this courageous step.”<br>“I'm honored to walk this journey with you.”<br>“I'm so supportive and proud of you.” | “Remember, consistency is key.”<br>“Remember that consistency is key.”<br>“One tip is to keep a food journal.” |

## no leaked chat marker and no degenerate text

| iter | cos(K0, K5) | more K=0 | more K=5 |
|---|---|---|---|
| 1 | 0.58 | “I couldn’t be more proud of you.”<br>“Thank you so much for your trust and openness.”<br>“I could not be more proud of you.” | “One tip is to keep a food journal.”<br>“One tip is to set small goals.”<br>“Here are three tips:” |
| 2 | 0.69 | “What can I do to support you?”<br>“Can you tell me a little bit about your diet?”<br>“Do you have any other concerns or questions?” | “It is called exposure therapy.”<br>“The possibilities are endless.”<br>“I’m talking about the mindset.” |
| 3 | 0.70 | “I'm thrilled to support you on your weight loss journey.”<br>“I'm excited to support you on your weight loss journey.”<br>“I'm so happy to support you on this journey.” | “What other ideas come to mind?”<br>“What could you focus on instead?”<br>“What comes to mind?” |
| 4 | 0.47 | “Awesome, I’m so proud of you!”<br>“Awesome, I'm so proud of you!”<br>“Great, I’m so proud of you!” | “Quitting smoking isn’t easy.”<br>“Sounds like you’re struggling with your smoking habit.”<br>“Nicotine replacement therapy, support groups, or self-help resources?” |
| 5 | 0.47 | “Awesome, I’m so proud of you!”<br>“Fantastic, I’m so proud of you!”<br>“Wonderful, I’m so proud of you!” | “Instead of focusing on”<br>“Instead of thinking about”<br>“So, instead of focusing on what you need to do,” |
| 6 | 0.10 | “Absolutely, I'm so proud of your commitment and enthusiasm.”<br>“Absolutely, I'm so proud of you!”<br>“I'm so proud of your commitment and progress!” | “That’s exactly what I was thinking.”<br>“That’s exactly what I’m thinking.”<br>“What comes to mind?” |
| 7 | 0.39 | “I’m always here to provide support and guidance.”<br>“I’m here to support you every step of the way.”<br>“I'm here to support you every step” | “It's about recognizing that”<br>“I think that sounds great.”<br>“That sounds great to me.” |
| 8 | 0.40 | “You have my full support and unconditional acceptance.”<br>“You have my full support and understanding.”<br>“You have my complete and unconditional support.” | “I’m talking about the way you think about your body.”<br>“Don’t just choose activities that you think will be fun.”<br>“What are your physical goals for yourself?” |
| 9 | -0.00 | “We're on the right track.”<br>“I see you’re right.”<br>“You’re hitting the nail on the head.” | “I started exercising, and my mood changed.”<br>“Track your cravings with an app.”<br>“Nicotine replacement therapy, support groups, or self-help resources?” |
| 10 | 0.10 | “I'm so proud of you for taking this courageous step.”<br>“I'm honored to walk this journey with you.”<br>“I’m so proud to be on this journey with you.” | “Remember, consistency is key.”<br>“Remember that consistency is key.”<br>“Are you interested in trying some of these tips?” |
