import React, { Component } from 'react';
import { Container } from 'reactstrap';
import RadialTree from '../RadialTree/RadialTree';
import Cell from "./Cell/Cell.js"
import './Image.css';

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

        window.onresize = this.getSize;
    }

    getSize(){
        this.setState({
            modal: true
        });
    }


    render() {
        this.labels=[];
        var c = 0;
        var color = "white";

        var h = window.innerHeight * 0.4;
        var w = window.innerHeight * 0.4;
        if(window.innerHeight*2 > window.innerWidth){
            w = window.innerWidth * 0.2;
            h = window.innerWidth * 0.2;
        }
        for(var i=0; i<this.props.colony.length; i++){
            var cell = this.props.colony[i];
            c = cell[0].length
            while(!this.props.colors.hasOwnProperty(cell[0].slice(1,c))){
                c--;
            }
            color = this.props.colors[cell[0].slice(1,c)];
            
            this.labels.push(
                <Cell
                colors={this.props.colors}
                key={cell[0]} k={cell[0]} color={color} 
                bottomR={window.innerHeight*0.5 + (Math.sin(this.props.angles[cell[0]])*this.props.pos*h)}
                leftR={window.innerWidth*0.75 + (Math.cos(this.props.angles[cell[0]])*this.props.pos*w)}
                top={(cell[2]*100/this.newImg.height).toString()+"%"}
                left={(cell[1]*100/this.newImg.width).toString()+"%"}/>);

        }
        if(this.state.modal){
            return (
                <div>
                    <RadialTree
                        curr={this.props.curr}
                        pos={this.props.pos}
                        frames={this.props.frames}
                        angles={this.props.angles}
                        colors={this.props.colors}/>
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

                </div>
            );
        }
        return null;
    }
}
