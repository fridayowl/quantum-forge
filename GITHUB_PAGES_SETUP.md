# GitHub Pages Deployment Guide

## 🚀 Enable GitHub Pages for QuantumForge

GitHub Pages needs to be enabled in your repository settings. Follow these steps:

### **Step 1: Go to Repository Settings**

1. Open your browser and go to: https://github.com/fridayowl/quantum-forge
2. Click on **Settings** (top right, near the repository name)

### **Step 2: Enable GitHub Pages**

1. In the left sidebar, scroll down and click on **Pages** (under "Code and automation")
2. Under **"Build and deployment"**, you'll see **"Source"**
3. Click the dropdown that says **"Deploy from a branch"**
4. Select **"GitHub Actions"** instead

### **Step 3: Verify Deployment**

1. The workflow should automatically run (or you can manually trigger it)
2. Go to the **Actions** tab in your repository
3. You should see "Deploy Web Demo to GitHub Pages" workflow running
4. Wait for it to complete (usually takes 1-2 minutes)
5. Once complete, your site will be live at: **https://fridayowl.github.io/quantum-forge**

---

## 🔧 Alternative: Quick Command Line Setup

If you prefer, you can also enable it via the web interface by visiting:
https://github.com/fridayowl/quantum-forge/settings/pages

---

## ✅ Verification

After enabling, you can verify the deployment:

```bash
# Check workflow status
gh run list --workflow="deploy-pages.yml"

# View the latest run
gh run view

# Open the deployed site
open https://fridayowl.github.io/quantum-forge
```

---

## 📝 What the Workflow Does

The GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) automatically:
1. Checks out the repository
2. Configures GitHub Pages
3. Uploads the `web-demo` directory as an artifact
4. Deploys it to GitHub Pages

---

## 🐛 Troubleshooting

### Issue: 404 Error
**Solution**: Make sure GitHub Pages is set to "GitHub Actions" as the source (Step 2 above)

### Issue: Workflow Fails
**Solution**: Check the Actions tab for error messages. The workflow needs the following permissions:
- `contents: read`
- `pages: write`
- `id-token: write`

These are already configured in the workflow file.

### Issue: Changes Not Showing
**Solution**: 
1. Clear your browser cache
2. Wait a few minutes for GitHub's CDN to update
3. Try accessing in an incognito/private window

---

## 🎉 Once Enabled

Your interactive quantum computing demo will be live at:
**https://fridayowl.github.io/quantum-forge**

Features available:
- 🔍 Grover's Search Algorithm
- ⚡ Circuit Optimizer
- 🔗 QAOA Max-Cut Solver
- 🧪 VQE Simulator (7 molecules!)
- 🌀 Quantum State Analyzer

---

## 🔄 Automatic Updates

Every time you push to the `master` branch, the workflow will automatically:
1. Rebuild the site
2. Deploy the latest version
3. Make it live within 1-2 minutes

No manual intervention needed after initial setup!
