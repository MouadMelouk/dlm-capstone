def test_single_image(image_path):
    # Load configuration
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    # Load weights
    if args.weights_path:
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print('===> Loaded checkpoint successfully!')
    else:
        print('Failed to load pre-trained weights')
        return

    # Set model to evaluation mode
    model.eval()

    # Run Grad-CAM on single image
    gradcam_image, explanation = test_single_image_GRADCAM(model, image_path)

    # Display the Grad-CAM image
    cv2.imshow("Grad-CAM", gradcam_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nFinal Output:")
    print(f"Explanation: {explanation}")

    return gradcam_image, explanation


# Example usage:
if __name__ == "__main__":
    image_path = "datasets/rgb/Celeb-DF-v1/Celeb-real/frames/id0_0000/000.png.png"  # Change this to your actual test image path
    test_single_image(image_path)
