import React, { Component } from 'react';
import { Container } from 'reactstrap';
import RadialTree from '../RadialTree/RadialTree';
import Cell from "./Cell/Cell.js"
import './Image.css';


const colors={
    "00": [1,0,0], // red
    "01": [0,1,0], // green
    "10": [0,0,1], // blue
    "11": [1,1,0]  // yellow
}

export default class ImageCell extends Component {
    constructor(props) {
        super(props);


        this.state = {
            modal: false
        };
        this.getSize = this.getSize.bind(this);
        this.labels = []

        this.newImg = new Image();
        this.newImg.onload = this.getSize;
        this.newImg.src = props.src;

    }

    getSize(){
        this.setState({
            modal: true
        });
    }

    render() {
        this.labels=[];
        // var hue = 0;
        var color = "";
        for(var i=2; i<this.props.colony.length; i++){
            var cell = this.props.colony[i];
            var code = cell[0][1]+cell[0][2];
            // hue  =  255-((this.props.colony[0]-(cell[0].length-1))/
            //         (this.props.colony[0]-this.props.colony[1]+1) * 255);
            // color = "rgb(" +
            //     (hue*colors[code][0]).toString() + "," +
            //     (hue*colors[code][1]).toString() + "," +
            //     (hue*colors[code][2]).toString() + ")";
            color = "rgb(" +
                (255*colors[code][0]).toString() + "," +
                (255*colors[code][1]).toString() + "," +
                (255*colors[code][2]).toString() + ")";
            this.labels.push(
                <div
                    key={cell[0]}
                    style={{
                        position:"absolute",
                        top: (cell[2]*100/this.newImg.height).toString()+"%",
                        left: (cell[1]*100/this.newImg.width).toString()+"%"
                        }}>
                    <Cell k={cell[0]} color={color}/>
                </div>)
        }
        if(this.state.modal){
            return (
                <div>
                    <Container  style={{
                                    margin:"0", 
                                    padding:"0",
                                    position:"relative",
                                    maxWidth:"50vw"
                                    }}>
                        <img
                            id="im"
                            src={this.props.src} 
                            alt="im"
                            className="image"
                            />
                        {this.labels}
                    </Container>

                    <RadialTree
                        pos={this.props.pos}
                        src={this.props.srcTree}
                        src_pie={this.props.src_pie}/>
                </div>
            );
        }
        return null;
    }
}
