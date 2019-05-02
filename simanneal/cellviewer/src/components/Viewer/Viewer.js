import React, { Component } from 'react';
import Sample from '../Sample/Sample.js';
import './Viewer.css';

export default class Viewer extends Component {
    constructor(props) {
        super(props);
        this.state = {
            modal: false
        };
        this.name="Sample Data";
        this.toggle = this.toggle.bind(this);
    }

    toggle() {
        this.setState(prevState => ({
            modal: !prevState.modal
        }));
    }

    render() {
        return (
            <Sample/>
        );
    }
}

