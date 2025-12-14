## README

It is important to note that this markdown has been made for the personal AI training model that Thomas and Watson have completed, not necessarily the runtime functionality of this code.

We are currently testing and improving the epoch mechanisms.

This repository allows for privatized AI real estate photographer generation to occur effectively. Check Google Drive on personal email for being able to download and run. It should work I guess. Just make sure you use a `python` environment. Install the latest dependencies.

### KEY NOTES

This is the documentation necessary for being able to:

1. Check latest epoch
2. Go to ~/LUTwithBGrid location
3. Run and deploy the latest model with the test data
4. See results in real time

```
find ~/LUTwithBGrid -name "_182_" -name "\*.pth"

for img in ~/Downloads/hackathon_validation/\*\_src.jpg; do
filename=$(basename "$img" \_src.jpg)
echo "Processing $filename with epoch 182..."
    python demo.py --input_path "$img" --output_path ~/LUTwithBGrid/watson_results/"${filename}\_epoch182.jpg" --pretrained_path saved_models/Bilateral_LUTs_sRGB/bilateral_lut_182.pth
done

mkdir -p ~/LUTwithBGrid/watson_results/epoch_182
mv ~/LUTwithBGrid/watson_results/\*\_epoch182.jpg ~/LUTwithBGrid/watson_results/epoch_182/

cd ~/LUTwithBGrid/watson_results/
zip -r epoch_182.zip epoch_182/

ls -la epoch_182.zip
```

Demo run functionality:

```
python demo.py --input_path ~/Downloads/hackathon_validation/10001_src.jpg --output_path ~/LUTwithBGrid/watson_results/10001_enhanced.jpg --pretrained_path pretrained/FiveK_sRGB.pth
```
