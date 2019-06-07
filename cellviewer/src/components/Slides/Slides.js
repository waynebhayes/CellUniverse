/*

Props:
- imgs      = [[imageSrc,framename]]    (main)
- srcTree   = tree SVG                  (radialtree)
- src_pie   = pie SVG                   (radialtree)
- colony    = {} from colony.JSON       (parseColony)
- angles    = {} from angles.JSON       (radialtree)

*/


import React, { Component } from 'react';
import { Button, Row } from 'reactstrap';
import ImageCell from './Image/Image';
import Slider from '@material-ui/lab/Slider';

export default class Slides extends Component {
    constructor(props) {
        super(props);
        this.state = {
            tog: 0
        };

        this.pos = 0;

        this.status = "play";

        this.images = props.imgs;
        this.colony = props.colony;
        this.change = this.change.bind(this);
        this.play = this.play.bind(this);
        this.click = this.click.bind(this);
        this.spaceBar = this.spaceBar.bind(this);
        window.onkeyup = this.spaceBar;
    }

    spaceBar(e){
        var key = e.keyCode ? e.keyCode : e.which;
        if (key === 32) {
            if (e.stopPropagation) {
                e.stopPropagation();
                e.preventDefault();
            }
            this.click();
        }
    }

    change(e,i) {
        if(i>=0 && i<this.images.length){
            this.pos = i;
            this.setState({
                tog : i
            });
        }
    }

    play(){
        if(this.status==="pause"){
            setTimeout(function() {
                this.change(null,(this.pos+1)%this.images.length);
                this.play();
            }.bind(this), 300);
        }
    }

    click(){
        if(this.status ==="pause"){
            this.status = "play"
            this.setState({
                tog : 0
            })
        }else{
            this.status = "pause"
            this.play()
        }
    }

    render() {
        if(this.images.length!==this.props.imgs.length){
            this.pos = 0;
        }
        this.images = this.props.imgs;
        this.colony = this.props.colony;
        return (
            <div>
                <Row>
                    <ImageCell
                        src={this.images[this.pos]} 
                        colony={this.colony[this.pos.toString()]["cells"]}
                        srcTree={this.props.srcTree}
                        pos={(this.pos+1.5)/(this.images.length+2)}
                        src_pie={this.props.src_pie}
                        angles={this.props.angles}/>
                    <Button
                        onClick={this.click}
                        style={{
                            width:"5%",
                            height:"3%",
                            position:"absolute",
                            bottom: "4.5%",
                            left: "3%"
                            }}
                    >
                        {this.status}
                    </Button>

                    <Slider
                        value={this.pos}
                        min={0}
                        max={this.images.length-1}
                        step={1}
                        onChange={this.change}
                        style={{
                            width:"80%",
                            height:"5%",
                            position:"absolute",
                            bottom: "1%",
                            right: "10%"
                        }}
                    />
                </Row>
            </div>
        );
    }
}
