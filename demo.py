from bark import generate_audio,preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk
import re
# nltk.download('punkt')
preload_models(text_use_gpu=True)

long_string1 = """
    TTS是一种先进的语音合成技术，它将文本转化为自然流畅的语音输出。TTS 技术的发展已经取得了显著的进展，
    """
long_string = """
    TTS是一种先进的语音合成技术，它将文本转化为自然流畅的语音输出。TTS 技术的发展已经取得了显著的进展，它在各种应用中都有广泛的应用，包括但不限于以下几个方面：

1. 辅助技术：TTS 技术对于视觉障碍者、文盲者以及有特殊需求的人士来说，提供了巨大的帮助。通过将书籍、网站内容和电子邮件等文本转化为语音，他们可以更轻松地获取信息。

2. 智能助手和虚拟助手：TTS 技术是智能助手和虚拟助手的重要组成部分。这些助手能够回答问题、提供信息，还能够以自然的方式与用户交流。

3. 自动导航系统：TTS 技术在GPS和导航系统中起着关键作用，它可以向司机提供路线指示，而无需将注意力从驾驶中分散出去。

4. 教育领域：TTS 技术可以帮助学生更好地理解教材内容，也可以支持有阅读障碍的学生。教育应用程序可以使用TTS技术，将课程材料朗读给学生，提供额外的学习支持。

5. 娱乐和媒体：TTS 技术已经在游戏、电影制作和广播领域取得了广泛应用。它可以用于为虚拟角色和电子游戏中的角色提供声音。

6. 多语言支持：TTS 技术可以轻松地将文本从一种语言翻译成另一种语言的语音。这对于跨文化交流和全球化市场至关重要。

7. 可访问性：对于老年人、残疾人士和那些不懂得阅读的人来说，TTS 技术提供了一种更容易获得信息的方式，增加了社会参与的机会。

TTS 技术的不断改进和发展意味着我们可以期待更加自然、流畅和逼真的语音合成，这将在各种领域继续产生积极的影响，并丰富人们的生活。未来，TTS 技术有望更好地模拟人类语音，提供更多的个性化选项，并在人机交互中发挥更重要的角色。
"""

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
# sentences = nltk.sent_tokenize(long_string)

sentences = cut_sent(long_string1)
# Set up sample rate
SAMPLE_RATE = 22050
HISTORY_PROMPT = "v2/zh_speaker_5"

chunks = ['']
token_counter = 0

for sentence in sentences:
	# print(sentence)
	current_tokens = len(nltk.Text(sentence))
	# print(current_tokens)
	if token_counter + current_tokens <= 100:
		token_counter = token_counter + current_tokens
		chunks[-1] = chunks[-1] + " " + sentence
	else:
		chunks.append(sentence)
		token_counter = current_tokens

# Generate audio for each prompt
audio_arrays = []

for prompt in chunks:
    audio_array = generate_audio(prompt,history_prompt=HISTORY_PROMPT,waveform_temp=1,text_temp=1)
    print(prompt + '\n')
    audio_arrays.append(audio_array)

# Combine the audio files
combined_audio = np.concatenate(audio_arrays)

# Write the combined audio to a file
write_wav("combined_audio.wav", SAMPLE_RATE, combined_audio)