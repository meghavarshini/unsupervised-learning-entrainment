#!/bin/bash
# List of Multicat trials with intact audio:
multicat_files=("T000627" "T000628" "T000631" "T000632"\
                "T000633" "T000634" "T000635" "T000636"\
                "T000637" "T000638" "T000671" "T000672"\
                "T000703" "T000704" "T000713" "T000714"\
                "T000715" "T000716" "T000727" "T000728"\
                "T000729" "T000730" "T000738")

# source directory for copying sound files:
source_dir="../../../../media/mule/projects/tomcat/protected/study-3_2022"

# Check if the source directory exists:
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' does not exist."
  exit 1
fi

# Target directory for copying sound files:
target_dir="../Asist3_data_management/multicat_complete_trials_data/"

# Check if the target directory exists
if [ ! -d "$target_dir" ]; then
  mkdir '$target_dir'
fi
echo "directories exist..."

# Loop through patterns and copy matching files
for p in "${multicat_files[@]}"; do
#    find "$source_dir" -type f -name "*$p*-E*.wav" -exec echo "Found file: {}" \;
  find "$source_dir" -type f -name "*$p*-E*.wav" -exec bash -c '
    file="$0"
    cp "$file" '"$target_dir"' && echo "File \"$file\" for pattern copied."
  ' {} \;
done

