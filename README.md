# TexModTools
- About
Some python scripts to help making texture mods for dx9 games with texMod and openTexMod

# Copyright Notice
 - TexModTools
 - First published 2022 - Copyright, Jakob Wapenhensch
 - Check LICENSE.md for full license text
 - https://creativecommons.org/licenses/by-nc/4.0/
 - https://creativecommons.org/licenses/by-nc/4.0/legalcode
  
# Updates
- 0.1 
  - Initial release


# Installation & Usage
- Place "DdsToPng.py" in the folder you use as output folder when extracting textures as dds with texMod.
- After extracting textures, start the script. It will convert any .dds file to .png while keeping the alpha channel.
- Place all textures you changed in a folder and make sure you keep the "_0X12345678.png" endings. The underscore and number are needed to be the right one to create .log files.
- Place "CreateLog.py" in the folder you put your changed textures in. Inside the file you will find a define "LOG_FILE_NAME". You can change that to the name you want for your .tpf file.
- Start the script. It will generate a .log file that can be used with TexMod to create a .tpf texture mod.
