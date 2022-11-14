# TexModTools
- About
Some python scripts to help making texture mods for dx9 games with texMod and openTexMod

# Copyright Notice
 - TexModTools
 - First published 2022 - Copyright, Jakob Wapenhensch
 - Check LICENSE.md for full license text
 - https://creativecommons.org/licenses/by-nc/4.0/
 - https://creativecommons.org/licenses/by-nc/4.0/legalcode
  

# Content 
- Original TexMod
- OpenTexMod beta_v1_r21.
- Python Script "DdsToPng.py"
- Python Script "CreateLog.py"

# Usage
This is just my personal workflow. I use 2 seperate versions of TexMod because every version i tried was broken or unfinished in some way.

- OpenTexMod is used to extract .dds files from the game.
- "DdsToPng.py" placed in the same folder as the extracted images, is used to convert all .dds files to.png files, while keeping the alpha channel.
- Now textures are edited via Photoshop or whatever and are placed in a seccond folder. The "_0X12345678.png" like file endings need to be kept intact from the underscore to the end!
- "CreateLog.py" placed in the same folder as the edited textures will create a .log file. Inside "CreateLog.py" you can find a variable called "LOG_FILE_NAME" which you can use to se the name of you .log file.
- Original TexMod is used to pack the .png's and the .log file to a .tpf texture pack.
# Updates
- 0.1 
  - Initial release
  - Added DdsToPng.py
  - Added CreateLog.py
- 0.2
  - Added OpenTexMod files, because their is no maintained download source.
  - Added original TexMod.exe for ease of use
  - Fixed some stuff


