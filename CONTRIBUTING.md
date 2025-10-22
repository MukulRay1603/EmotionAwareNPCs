# Contributing to Emotion-Aware NPCs

Thank you for your interest in contributing to the Emotion-Aware NPCs project! This document provides guidelines and information for team members and contributors.

## Team Structure

### Phase 1 Team Members
- **P1 (Rahul Sharma)**: CV Lead - FER baseline, core CV modules
- **P2 (Mukul Ray)**: Unity Lead - Webcam capture, UI, game logic  
- **P3 (Karthik Ramanathan)**: Data & Eval Lead - Datasets, user study, robustness tests
- **P4 (Taner Bulbul)**: MLOps & Release Lead - Repo management, CI/CD, ONNX pipeline

## Development Workflow

### Branch Strategy
- `main`: Production-ready code, stable releases
- `develop`: Integration branch for ongoing development
- `feature/*`: Feature branches for specific tasks
- `hotfix/*`: Critical bug fixes

### Getting Started

1. **Fork the repository** (if external contributor)
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EmotionAwareNPCs.git
   cd EmotionAwareNPCs
   ```

3. **Set up remote**:
   ```bash
   git remote add upstream https://github.com/MukulRay1603/EmotionAwareNPCs.git
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Guidelines

#### Code Style
- **Python**: Follow PEP 8, use type hints
- **C# (Unity)**: Follow Microsoft C# conventions
- **Documentation**: Use clear, concise comments

#### Commit Messages
Use conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(cv): add MobileNetV2 FER model`
- `fix(backend): resolve CORS issue`
- `docs(readme): update setup instructions`

#### Pull Request Process

1. **Create PR** from your feature branch to `develop`
2. **Fill out PR template**:
   - Description of changes
   - Testing performed
   - Screenshots/videos (if applicable)
   - Related issues

3. **Request review** from relevant team members:
   - P1 for CV-related changes
   - P2 for Unity-related changes
   - P3 for data/evaluation changes
   - P4 for infrastructure changes

4. **Address feedback** and make requested changes
5. **Merge** after approval and CI passes

## Component-Specific Guidelines

### Computer Vision (P1 - Rahul)
- **Location**: `cv/` directory
- **Key files**: `inference/`, `models/`, `preprocessing/`
- **Requirements**:
  - Models must be ONNX format
  - Include inference scripts
  - Add unit tests for core functions
  - Document model performance metrics

### Unity Client (P2 - Mukul)
- **Location**: `unity/` directory
- **Key files**: `Assets/Scripts/`, `Assets/Scenes/`
- **Requirements**:
  - Test on Unity 2022.3 LTS
  - Include scene files
  - Document setup instructions
  - Record demo videos

### Data & Evaluation (P3 - Karthik)
- **Location**: `docs/phase1_report/` directory
- **Key files**: Dataset summaries, evaluation metrics
- **Requirements**:
  - Document dataset statistics
  - Include baseline performance metrics
  - Create evaluation reports
  - Maintain data versioning

### MLOps & Integration (P4 - Taner)
- **Location**: `backend/` directory
- **Key files**: `main.py`, API endpoints
- **Requirements**:
  - Maintain API documentation
  - Ensure CI/CD pipeline works
  - Monitor performance metrics
  - Handle deployment issues

## Testing Requirements

### Unit Tests
- **Python**: Use pytest
- **C#**: Use Unity Test Framework
- **Coverage**: Aim for >80% code coverage

### Integration Tests
- Test API endpoints
- Test Unity-API communication
- Test end-to-end pipeline

### Performance Tests
- Measure inference latency
- Test frame rate performance
- Validate memory usage

## Documentation Standards

### Code Documentation
- **Functions**: Docstrings with parameters and return values
- **Classes**: Purpose and usage examples
- **Complex logic**: Inline comments

### API Documentation
- Use FastAPI automatic documentation
- Include example requests/responses
- Document error codes and handling

### README Files
- Clear setup instructions
- Prerequisites and dependencies
- Usage examples
- Troubleshooting section

## Issue Management

### Bug Reports
Use the bug report template:
```
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, macOS 12.0]
- Python version: [e.g., 3.8.10]
- Unity version: [e.g., 2022.3.0f1]

**Screenshots/Logs**
If applicable, add screenshots or error logs
```

### Feature Requests
Use the feature request template:
```
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## Release Process

### Phase 1 (Current)
- **Target**: October 22, 2025
- **Deliverables**: Working prototype
- **Criteria**: End-to-end pipeline functional

### Future Phases
- **Phase 2**: Enhanced features and performance
- **Phase 3**: Production deployment
- **Phase 4**: Advanced ML features

## Communication

### Team Communication
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and discussion
- **GitHub Discussions**: General questions and ideas

### Meeting Schedule
- **Weekly Standup**: Fridays 2:00 PM EST
- **Sprint Planning**: Every 2 weeks
- **Retrospectives**: End of each phase

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Be respectful and constructive
- Focus on what's best for the community
- Show empathy towards other team members
- Accept constructive criticism gracefully

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks or political discussions
- Spam or off-topic discussions

## Getting Help

### Documentation
- Check existing documentation first
- Look for similar issues in GitHub
- Review code comments and examples

### Team Support
- Tag relevant team members in issues
- Use @mentions for specific expertise
- Ask questions in GitHub Discussions

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Unity Documentation](https://docs.unity3d.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and team members
- Special thanks to the open-source community
- Inspiration from existing emotion recognition research

---

**Last Updated**: October 21, 2025  
**Version**: 1.0.0  
**Maintainer**: Taner Bulbul (P4 - MLOps Lead)
