@ECHO on
SETLOCAL
set VIDEO_DIR="E:/work/big-project/VideoSummarization/dataset/dataset_input/Cricket_Dataset"
set VIDEO_DIR="E:/work/big-project/VideoSummarization/dataset/dataset_input"

set AUDIO_DIR="E:/work/big-project/VideoSummarization/dataset/dataset_audio"
set OUTPUT_DIR="E:/work/big-project/VideoSummarization/dataset/output"
mkdir "%AUDIO_DIR%"
CD %VIDEO_DIR%
%VIDEO_DIR:~0,2%
FOR %%i IN (*) DO (
set "filedrive=%%~di"
set "filepath=%%~pi"
set "filename=%%~ni"
set "fileextension=%%~xi"

IF EXIST "%OUTPUT_DIR%/%%~ni%%~xi" (
echo Yes 
) ELSE (
mkdir "%OUTPUT_DIR%/%%~ni%%~xi"
echo "==============================Extracting Audio Frames =============================="
#ffmpeg -i videoplayback_cricket.3gp -vn -acodec pcm_s16le -ar 44100 -ac 1 output.wav
ffmpeg -i "%%~ni%%~xi" -vn -n -acodec pcm_s16le -ar 44100 -ac 1 "%AUDIO_DIR%/%%~ni%%~xi.wav"
#ffmpeg -i "%%~ni%%~xi" -vn -n -acodec pcm_s16le -ar 44100 -ac 2 "%AUDIO_DIR%/%%~ni%%~xi.wav"
echo "==============================Extracting Video Frames =============================="
ffmpeg -i "%%~ni%%~xi" -n -r 20/1 "%OUTPUT_DIR%/%%~ni%%~xi/%%d.png"
)


)
